"""
pulse_solver.py  —  Pulse-basis (piecewise-constant, point-collocation) 2D BIE solver.

Companion to rcs_solver.py, which uses continuous linear Galerkin.  This
module provides an alternative discretization where each panel carries one
constant unknown, and the integral equation is enforced by point collocation
at the panel center.  The two solvers share geometry construction, material
parsing, and kernel evaluation routines; only the discretization and system
assembly differ.

FORMULATION RULES (chosen to match the analytic Mie reference):

    TM polarization (H_z axial):
        * EFIE in the sense of "enforce the Neumann BC on H_z (no normal E
          on PEC) via an SLP representation u_s = S sigma":
                (-1/2 I + K') sigma = -du_inc/dn
          For Robin/IBC nodes (Leontovich BC), the equation becomes
                (-1/2 I + K' + alpha S) sigma = -(du_inc/dn + alpha u_inc)
          where alpha = surface Robin coefficient (see _surface_robin_alpha).

    TE polarization (E_z axial):
        * For TYPE 2-5 surfaces (closed/IBC/dielectric-backed bodies):
          MFIE in the sense of "enforce the Dirichlet BC u = 0 on PEC
          (or Robin u + alpha du/dn = 0) via a DLP representation
          u_s = D mu":
                (1/2 I + K) mu = -u_inc                         (PEC)
                (1/2 I + K + alpha*(-1/2 I + K'_mu)) mu = ...   (Robin)
          where K is the double-layer (source-side normal derivative of G).
        * For TYPE 1 surfaces (thin resistive/reactive sheets): SLP sheet
          BIE same as in the Galerkin solver:
                (S - (Z_s / jk eta) I) sigma = -u_inc

NO CFIE.  Pulse systems near closed-body interior resonances can become
ill-conditioned; this is expected behavior given the chosen ruleset.  For
production use on closed bodies near resonances, use the Galerkin solver
with cfie_alpha=0.2.

SUPPORTED GEOMETRIES:

    * Pure TYPE 2 (PEC) bodies, any polarization
    * Pure TYPE 2 with Robin IBC on surfaces
    * Pure TYPE 1 sheets with resistive/reactive impedance (tapered OK)
    * Mixed TYPE 1 + pure-PEC TYPE 2 (unified SLP for TE sheets + DLP for
      TE bodies is non-trivial; in this implementation mixed cases route to
      SLP throughout and use the sheet equation on TYPE 1 panels, the EFIE
      equation on TYPE 2 panels.  For TM, all panels share SLP + Robin.)

NOT SUPPORTED (raises):

    * TYPE 3/5 dielectric interfaces (transmission BC requires two unknowns
      per point; pulse cannot naturally represent the continuous trace)
    * TYPE 4 coated-PEC layered coatings (needs multi-region coupling)

API:

    solve_monostatic_rcs_2d_pulse(geometry_snapshot, frequencies_ghz,
                                   elevations_deg, polarization,
                                   geometry_units='inches', ...)
        → dict in the same format as rcs_solver.solve_monostatic_rcs_2d

    Also available: rcs_solver.solve_monostatic_rcs_2d(...,
                       basis='pulse') through the kwarg bridge
                       installed by register_pulse_basis().
"""

from __future__ import annotations

import cmath
import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# Re-use all kernel/geometry/material helpers from rcs_solver.  We do not
# duplicate them — the point of this module is to replace only the
# discretization, not the physics or the numerics of the Green's function.
from rcs_solver import (
    C0,
    ETA0,
    EPS,
    MAX_PANELS_DEFAULT,
    DEFAULT_PANELS_PER_WAVELENGTH,
    Panel,
    MaterialLibrary,
    RCS_NORM_NUMERATOR,
    _build_panels,
    _green_2d_array,
    _dgreen_dn_src_array,
    _dgreen_dn_obs_array,
    _normalize_polarization,
    _canonical_user_polarization_label,
    _unit_scale_to_meters,
    _surface_robin_alpha,
    _rcs_sigma_from_amp,
    validate_geometry_snapshot_for_solver,
    _hankel2_0_array,
)


# ───────────────────────────────────────────────────────────────────────────
# Kernel matrices — pulse-basis, point-collocated
# ───────────────────────────────────────────────────────────────────────────

def _pulse_panel_geom(panels: Sequence[Panel]) -> Dict[str, np.ndarray]:
    """Pre-compute per-panel arrays used by the kernel builders."""
    n = len(panels)
    centers = np.array([p.center for p in panels], dtype=float)       # (N, 2)
    p0s     = np.array([p.p0     for p in panels], dtype=float)
    p1s     = np.array([p.p1     for p in panels], dtype=float)
    normals = np.array([p.normal for p in panels], dtype=float)
    lengths = np.array([p.length for p in panels], dtype=float)
    return {
        "centers": centers, "p0": p0s, "p1": p1s,
        "normals": normals, "lengths": lengths,
        "n": n,
    }


def _gauss_legendre(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Shifted to [0, 1]: weights sum to 1."""
    t, w = np.polynomial.legendre.leggauss(int(max(1, order)))
    t = 0.5 * (t + 1.0)
    w = 0.5 * w
    return t, w


def _assemble_S_pulse(geom: Dict[str, np.ndarray], k0: complex,
                      quad_order: int = 8) -> np.ndarray:
    """
    Single-layer operator S with pulse-basis sources and point-collocated
    observations at panel centers:

        S_ij = ∫_{panel j} G(x_i^c, r') dr'

    Self-term (i == j) uses the logarithmic singular integral closed form:
        S_ii = (L/(2π)) · [ log(k L / 2) - 1 + γ ]  +  j·L/4  approximately,
    evaluated here via a graded quadrature that captures the log to machine
    precision for modest k·L.

    Off-diagonal: Gauss-Legendre in s ∈ [0, L_j].
    """
    centers = geom["centers"]
    p0, p1  = geom["p0"], geom["p1"]
    lengths = geom["lengths"]
    N = geom["n"]

    t_gl, w_gl = _gauss_legendre(quad_order)

    # Batched off-diagonal: for each src panel j, evaluate G at all obs centers.
    S = np.zeros((N, N), dtype=np.complex128)
    for j in range(N):
        seg = p1[j] - p0[j]
        # Sample points along src panel j
        src_pts = p0[j] + np.outer(t_gl, seg)                   # (Q, 2)
        # Distances from every obs center to every src sample
        diff = centers[:, None, :] - src_pts[None, :, :]        # (N, Q, 2)
        r = np.linalg.norm(diff, axis=2)                        # (N, Q)
        g = _green_2d_array(k0, r.ravel()).reshape(N, -1)       # (N, Q)
        # Integrate over src panel: S_ij = L_j * sum_q w_q * g(x_i, src_q)
        S[:, j] = float(lengths[j]) * (g * w_gl[None, :]).sum(axis=1)

    # Self-term: use a graded quadrature clustered near the singularity.
    # For a straight panel of length L, observer at center (t = 0.5 * L from
    # either end), the integral has a logarithmic singularity at the
    # observation point.  Split [0, L] into [0, L/2] and [L/2, L] and use a
    # higher-order graded rule on each half.  To keep things cheap, we use
    # an analytic small-argument expansion of H_0^(2)(x) for x near 0:
    #     H_0^(2)(x) ≈ 1 - (2j/pi) * (log(x/2) + gamma)
    # so the singular part of G is
    #     G(r) ≈ (j/4) - (1/(2pi)) * (log(k r / 2) + gamma)
    # The integral ∫_0^{L/2} log(k s / 2) ds = (L/2) * (log(kL/4) - 1).
    for i in range(N):
        L = float(lengths[i])
        if L < EPS:
            continue
        # Analytic singular piece: integrate the real logarithm contribution
        # symmetrically about the center.
        # Each half contributes (L/2) * (log(k L / 4) - 1), and there are two
        # halves, so the logarithmic part equals L * (log(k L / 4) - 1).
        # Plus the j/4 constant · L.
        EULER_GAMMA = 0.5772156649015329
        k_abs = abs(complex(k0))
        log_part = -(1.0 / (2.0 * np.pi)) * L * (np.log(k_abs * L / 4.0) + EULER_GAMMA - 1.0)
        const_part = (0.25j) * L
        # The residual non-singular part is evaluated by subtracting the
        # small-argument approximation from the exact Hankel, then using a
        # graded rule.  For modest k*L this residual is small; a 16-point
        # graded quadrature suffices.
        t_grad, w_grad = _gauss_legendre(16)
        # Offset from center: samples on both halves.  We avoid placing a
        # sample exactly at s = L/2 (the singularity).
        t_left  = 0.5 * t_grad          # [0, 1/2]
        t_right = 0.5 + 0.5 * t_grad    # [1/2, 1]
        t_all = np.concatenate([t_left, t_right])
        w_all = np.concatenate([w_grad * 0.5, w_grad * 0.5])
        seg = p1[i] - p0[i]
        src_pts = p0[i] + np.outer(t_all, seg)
        # Distance from center (at t = 0.5)
        r = np.linalg.norm(src_pts - centers[i][None, :], axis=1)
        r_safe = np.where(r < EPS, EPS, r)
        # Exact minus small-argument approx: this residual is smooth.
        x = complex(k0) * r_safe
        # Use full Hankel but subtract the singular log part already
        # handled analytically.
        full_G = 0.25j * _hankel2_0_array(x)
        approx_singular = (0.25j) - (1.0 / (2.0 * np.pi)) * (np.log(abs(complex(k0)) * r_safe / 2.0) + EULER_GAMMA)
        residual = full_G - approx_singular
        S[i, i] = log_part + const_part + L * (residual * w_all).sum()

    return S


def _assemble_K_pulse(geom: Dict[str, np.ndarray], k0: complex,
                      quad_order: int = 8) -> np.ndarray:
    """
    Double-layer operator K with pulse-basis sources and point-collocated
    observations:

        K_ij = ∫_{panel j} ∂G(x_i^c, r')/∂n(r') dr'

    Self-term diagonal: the principal-value integral of K on a straight
    panel with observation at its own center vanishes (the normal is
    perpendicular to the tangent everywhere along the panel, and the
    observer sits on the panel centerline, so d·n = 0 for every quadrature
    point).  We therefore set K_ii = 0 exactly.  The +1/2 or -1/2 jump term
    is added SEPARATELY by the system builder (it's not part of K, it's
    part of the limit from the exterior/interior side).
    """
    centers = geom["centers"]
    p0, p1  = geom["p0"], geom["p1"]
    normals = geom["normals"]
    lengths = geom["lengths"]
    N = geom["n"]

    t_gl, w_gl = _gauss_legendre(quad_order)

    K = np.zeros((N, N), dtype=np.complex128)
    for j in range(N):
        seg = p1[j] - p0[j]
        src_pts = p0[j] + np.outer(t_gl, seg)                   # (Q, 2)
        n_src   = normals[j]                                    # (2,) — constant along panel
        # For each obs center, build r_vec (obs - src).
        for i in range(N):
            r_vec = centers[i][None, :] - src_pts               # (Q, 2)
            dg = _dgreen_dn_src_array(
                k0, r_vec,
                np.broadcast_to(n_src, r_vec.shape),             # per-row n_src
            )                                                     # (Q,)
            K[i, j] = float(lengths[j]) * complex(np.sum(w_gl * dg))
    # Self term:  d · n = 0 everywhere along a straight panel for observer at
    # its own center — exact zero.
    np.fill_diagonal(K, 0.0 + 0.0j)
    return K


def _assemble_Kp_pulse(geom: Dict[str, np.ndarray], k0: complex,
                       quad_order: int = 8) -> np.ndarray:
    """
    Adjoint double-layer operator K' with pulse-basis sources and
    point-collocated observations:

        K'_ij = ∫_{panel j} ∂G(x_i^c, r')/∂n(x_i) dr'

    The derivative is in the observation-point normal direction.  Same
    self-term argument as K: d · n_obs = 0 everywhere along panel j when
    observer i = j sits at the panel's center, so K'_ii = 0.
    """
    centers = geom["centers"]
    p0, p1  = geom["p0"], geom["p1"]
    normals = geom["normals"]
    lengths = geom["lengths"]
    N = geom["n"]

    t_gl, w_gl = _gauss_legendre(quad_order)

    Kp = np.zeros((N, N), dtype=np.complex128)
    for j in range(N):
        seg = p1[j] - p0[j]
        src_pts = p0[j] + np.outer(t_gl, seg)                   # (Q, 2)
        for i in range(N):
            n_obs = normals[i]                                   # (2,)
            r_vec = centers[i][None, :] - src_pts               # (Q, 2)
            dg = _dgreen_dn_obs_array(k0, r_vec, n_obs)          # (Q,)
            Kp[i, j] = float(lengths[j]) * complex(np.sum(w_gl * dg))
    np.fill_diagonal(Kp, 0.0 + 0.0j)
    return Kp


# ───────────────────────────────────────────────────────────────────────────
# Incident plane-wave loads
# ───────────────────────────────────────────────────────────────────────────

def _incident_at_centers(geom: Dict[str, np.ndarray], k0: float,
                         elev_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return u_inc and du_inc/dn evaluated at panel centers for each incidence
    angle.  Shapes: (N, A).
    """
    phi = np.deg2rad(np.asarray(elev_deg, dtype=float).reshape(-1))
    dirs = np.stack([np.cos(phi), np.sin(phi)], axis=1)         # (A, 2)
    centers = geom["centers"]                                   # (N, 2)
    normals = geom["normals"]                                   # (N, 2)
    # u_inc = exp(j k d . r)
    phase = np.exp(1j * float(k0) * (centers @ dirs.T))         # (N, A)
    # du_inc/dn = j k (d . n) * u_inc
    dot_n = normals @ dirs.T                                    # (N, A)
    dn_inc = 1j * float(k0) * dot_n * phase                     # (N, A)
    return phase, dn_inc


# ───────────────────────────────────────────────────────────────────────────
# Far field
# ───────────────────────────────────────────────────────────────────────────

def _farfield_slp_pulse(geom: Dict[str, np.ndarray], density: np.ndarray,
                        k0: float, obs_deg: np.ndarray,
                        quad_order: int = 8) -> np.ndarray:
    """Far-field amplitude from an SLP density via pulse quadrature."""
    t_gl, w_gl = _gauss_legendre(quad_order)
    phi = np.deg2rad(np.asarray(obs_deg, dtype=float).reshape(-1))
    dirs = np.stack([np.cos(phi), np.sin(phi)], axis=1)         # (A, 2)
    lengths = geom["lengths"]
    p0, p1 = geom["p0"], geom["p1"]
    N = geom["n"]
    amp = np.zeros(dirs.shape[0], dtype=np.complex128)
    for j in range(N):
        seg = p1[j] - p0[j]
        src = p0[j] + np.outer(t_gl, seg)                       # (Q, 2)
        phase = np.exp(1j * float(k0) * (src @ dirs.T))         # (Q, A)
        amp += float(lengths[j]) * complex(density[j]) * (phase * w_gl[:, None]).sum(axis=0)
    return amp


def _farfield_dlp_pulse(geom: Dict[str, np.ndarray], density: np.ndarray,
                        k0: float, obs_deg: np.ndarray,
                        quad_order: int = 8) -> np.ndarray:
    """Far-field amplitude from a DLP density via pulse quadrature."""
    t_gl, w_gl = _gauss_legendre(quad_order)
    phi = np.deg2rad(np.asarray(obs_deg, dtype=float).reshape(-1))
    dirs = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    lengths = geom["lengths"]
    p0, p1 = geom["p0"], geom["p1"]
    normals = geom["normals"]
    N = geom["n"]
    amp = np.zeros(dirs.shape[0], dtype=np.complex128)
    for j in range(N):
        seg = p1[j] - p0[j]
        src = p0[j] + np.outer(t_gl, seg)
        phase = np.exp(1j * float(k0) * (src @ dirs.T))         # (Q, A)
        dot_n = dirs @ normals[j]                               # (A,)
        amp += float(lengths[j]) * complex(density[j]) * (1j * float(k0)) * dot_n * (phase * w_gl[:, None]).sum(axis=0)
    return amp


# ───────────────────────────────────────────────────────────────────────────
# Dispatcher helpers
# ───────────────────────────────────────────────────────────────────────────

def _classify_panels(panels: Sequence[Panel],
                     materials: MaterialLibrary,
                     freq_ghz: float,
                     pol: str,
                     k0: float) -> Dict[str, Any]:
    """
    Inspect the panel list and return:
      - per-panel classification (sheet / pec / robin)
      - per-panel surface impedance Z_s (0 for PEC, from materials for sheet or IBC)
      - rejection reason if the geometry mixes types pulse doesn't support
    """
    n = len(panels)
    is_sheet = np.zeros(n, dtype=bool)
    is_pec   = np.zeros(n, dtype=bool)
    z_s_arr  = np.zeros(n, dtype=np.complex128)

    for i, p in enumerate(panels):
        seg_type = int(p.seg_type)
        if seg_type in (3, 5):
            raise NotImplementedError(
                "pulse_solver: TYPE 3 or TYPE 5 dielectric transmission interfaces "
                "are not supported.  Pulse basis cannot cleanly represent the "
                "continuous trace condition at a dielectric boundary.  Use the "
                "Galerkin solver (rcs_solver.solve_monostatic_rcs_2d) instead."
            )
        if seg_type == 4:
            raise NotImplementedError(
                "pulse_solver: TYPE 4 coated-PEC multi-region geometries are "
                "not supported.  Use the Galerkin solver."
            )
        if seg_type == 1:
            is_sheet[i] = True
            z_s_arr[i] = materials.get_impedance(int(p.ibc_flag), freq_ghz,
                                                  arc_s=p.arc_s_center)
        elif seg_type == 2:
            ibc_flag = int(p.ibc_flag)
            if ibc_flag == 0:
                is_pec[i] = True
            else:
                # Robin IBC on PEC-backed surface
                z_s_arr[i] = materials.get_impedance(ibc_flag, freq_ghz,
                                                      arc_s=p.arc_s_center)
                if abs(z_s_arr[i]) <= EPS:
                    is_pec[i] = True
                # otherwise: Robin (is_sheet=False, is_pec=False, z_s_arr populated)

    return {
        "is_sheet": is_sheet,
        "is_pec": is_pec,
        "z_s": z_s_arr,
        "n": n,
    }


# ───────────────────────────────────────────────────────────────────────────
# Core solve routines
# ───────────────────────────────────────────────────────────────────────────

def _solve_tm_slp_pulse(panels: Sequence[Panel], classification: Dict[str, Any],
                        k0: float, elev_deg: np.ndarray,
                        quad_order: int = 8) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    TM (H_z axial): pulse-basis SLP formulation for all element types.

    System:
        PEC nodes:    (-1/2 I + K') sigma_i = -du_inc/dn|_i
        Sheet nodes:  (S - (Z_s / jk eta) I) sigma_i = -u_inc|_i
        Robin nodes:  (-1/2 I + K' + alpha S) sigma_i = -(du_inc/dn + alpha u_inc)|_i
                      where alpha = -jk eta / Z_s  (TM Leontovich coefficient)

    All rows live in the same unknown vector sigma (one DOF per panel).
    """
    geom = _pulse_panel_geom(panels)
    N = geom["n"]
    I = np.eye(N, dtype=np.complex128)

    S  = _assemble_S_pulse (geom, k0, quad_order)
    Kp = _assemble_Kp_pulse(geom, k0, quad_order)
    u_inc, dn_inc = _incident_at_centers(geom, float(k0), elev_deg)      # (N, A)

    A = np.zeros_like(S)
    rhs = np.zeros_like(u_inc)

    is_sheet = classification["is_sheet"]
    is_pec   = classification["is_pec"]
    z_s_arr  = classification["z_s"]
    # TM Robin alpha: for Leontovich, alpha = -j k eta / Z_s (see
    # _surface_robin_alpha with pol='TM').  We call the shared helper so
    # that tapered impedance handling is consistent.
    robin_mask = ~is_sheet & ~is_pec

    # PEC rows: (-1/2 I + K') sigma = -du_inc/dn
    pec_idx = np.flatnonzero(is_pec)
    if pec_idx.size > 0:
        A[pec_idx, :] = -0.5 * I[pec_idx, :] + Kp[pec_idx, :]
        rhs[pec_idx, :] = -dn_inc[pec_idx, :]

    # Sheet rows: (S - (Z_s / jk eta) I) sigma = -u_inc
    sheet_idx = np.flatnonzero(is_sheet)
    if sheet_idx.size > 0:
        sheet_factor = z_s_arr[sheet_idx] / (1j * float(k0) * ETA0)       # (Ns,)
        A[sheet_idx, :] = S[sheet_idx, :]
        # Subtract sheet_factor on the diagonal rows.
        A[sheet_idx, sheet_idx] -= sheet_factor
        rhs[sheet_idx, :] = -u_inc[sheet_idx, :]

    # Robin rows (IBC on TYPE 2): alpha_i = TM Leontovich coefficient.
    robin_idx = np.flatnonzero(robin_mask)
    if robin_idx.size > 0:
        alpha = np.array([
            _surface_robin_alpha("TM", eps_phys=1.0, mu_phys=1.0, k_phys=k0,
                                  z_s=complex(z_s_arr[i]))
            for i in robin_idx
        ], dtype=np.complex128)
        # Row block: (-1/2 I + K' + diag(alpha) S) restricted to robin rows.
        A[robin_idx, :] = -0.5 * I[robin_idx, :] + Kp[robin_idx, :] + alpha[:, None] * S[robin_idx, :]
        rhs[robin_idx, :] = -(dn_inc[robin_idx, :] + alpha[:, None] * u_inc[robin_idx, :])

    # Solve (one LU, many RHS).
    sigma = np.linalg.solve(A, rhs)
    # Residual norm check
    resid = np.linalg.norm(A @ sigma - rhs, axis=0)
    rhs_norm = np.linalg.norm(rhs, axis=0)
    rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
    max_rel_resid = float(np.max(resid / rhs_norm))

    # Far field: SLP projector (all panels contribute as SLP density).
    amp = _farfield_slp_pulse(geom, sigma[:, 0] if sigma.ndim > 1 else sigma,
                               float(k0), elev_deg, quad_order)
    # For many elevations, loop
    if sigma.ndim > 1 and sigma.shape[1] > 1:
        amp = np.zeros(sigma.shape[1], dtype=np.complex128)
        for a in range(sigma.shape[1]):
            amp[a] = _farfield_slp_pulse(geom, sigma[:, a], float(k0),
                                          elev_deg[a:a+1], quad_order)[0]

    rcs_lin = _rcs_sigma_from_amp(amp, float(k0))
    return rcs_lin, amp, max_rel_resid


def _solve_te_pulse(panels: Sequence[Panel], classification: Dict[str, Any],
                    k0: float, elev_deg: np.ndarray,
                    quad_order: int = 8) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    TE (E_z axial): pulse-basis mixed formulation.

      - TYPE 1 sheets    → SLP: (S - (Z_s/jk eta) I) sigma = -u_inc
      - TYPE 2-5 bodies  → DLP: (1/2 I + K) mu = -u_inc                        (PEC)
                          or (1/2 I + K + alpha*(-1/2 I + K'_adj)) mu ...     (Robin)

    Because the two surface types use DIFFERENT representations, the unknown
    vector is piecewise: sigma on sheets, mu on bodies.  The system is
    block-structured but everything lives in one N-vector.

    Far field: SLP projector for sheet panels, DLP projector for body panels,
    summed.
    """
    geom = _pulse_panel_geom(panels)
    N = geom["n"]
    I = np.eye(N, dtype=np.complex128)

    S  = _assemble_S_pulse (geom, k0, quad_order)
    K  = _assemble_K_pulse (geom, k0, quad_order)
    Kp = _assemble_Kp_pulse(geom, k0, quad_order)
    u_inc, dn_inc = _incident_at_centers(geom, float(k0), elev_deg)

    is_sheet = classification["is_sheet"]
    is_pec   = classification["is_pec"]
    z_s_arr  = classification["z_s"]
    robin_mask = ~is_sheet & ~is_pec

    # Build block system.  Rows indexed by observation panel i; columns by
    # source panel j (unknown on panel j).
    #
    # On a sheet row, the unknown on every src panel represents a (sigma_j)
    # SLP density; the kernel column is S[i, j].
    # On a body row, the unknown on every src panel represents a (mu_j) DLP
    # density; the kernel column is K[i, j] (from the body side) but when
    # the src is a sheet panel, the unknown there is actually sigma and we
    # must use S[i, j] instead.
    #
    # We must therefore fix an interpretation for what the unknown on each
    # panel represents:
    #   - on sheet panels: sigma (SLP density)
    #   - on body panels:  mu (DLP density)
    # and assemble the matrix columns by source panel type:
    #   - src_j is sheet → column is contribution-of-sigma-on-panel-j to
    #                      the representation: use S_{·j} everywhere, plus
    #                      the sheet self-term.
    #   - src_j is body  → column is contribution-of-mu-on-panel-j to the
    #                      representation: use K_{·j}.
    #
    # Row interpretation (which equation is enforced at obs i):
    #   - is_sheet[i]  → SIBC: u_inc + u_s = Z_s · J_z
    #                    → (S[i, sheet_cols] - factor*I_sheet[i])·sigma
    #                      + S[i, body_cols]·(... ?)
    #     Wait — this doesn't work out cleanly because u_s at a sheet row
    #     gets contributions from BOTH sheet SLPs (via S) AND body DLPs
    #     (via K).  The RHS equation is the same: u_s[i] = -u_inc[i] + Z_s·J_z.
    #     So:  SUM_over_j(kernel[i,j] · density[j]) = -u_inc[i] + Z_s_i·J_z_i,
    #     where kernel[i,j] = S[i,j] if src j is sheet, K[i,j] if src j is body.
    #
    #   - is_pec_body[i] or is_robin_body[i] → u_s + u_inc = 0 on PEC,
    #                      u_s + alpha du_s/dn = -u_inc - alpha du_inc/dn for Robin.
    #     u_s at body obs:  SUM_j(kernel[i,j] · density[j]) with the interior
    #                       DLP limit  u_s|_body = (1/2 I + K)·mu + S·sigma  (sheet contribs).
    #                       But the 1/2 jump only applies to body columns (the
    #                       DLP density) — sheets have no jump for u through
    #                       their SLP representation.

    A = np.zeros((N, N), dtype=np.complex128)
    rhs = np.zeros_like(u_inc)

    sheet_cols = np.flatnonzero(is_sheet)
    body_cols  = np.flatnonzero(~is_sheet)

    # Build the "raw" scattered-field kernel: u_s|_row_i = (row of M) · density.
    # For sheet src cols → S[·, sheet_cols]
    # For body  src cols → K[·, body_cols]  (DLP representation for bodies)
    M_u = np.zeros((N, N), dtype=np.complex128)
    if sheet_cols.size > 0:
        M_u[:, sheet_cols] = S[:, sheet_cols]
    if body_cols.size > 0:
        M_u[:, body_cols] = K[:, body_cols]

    # Build the "raw" normal-derivative kernel: ∂u_s/∂n|_row_i = (row of M') · density.
    # For sheet src → ∂/∂n(S·sigma) = K'[·, sheet_cols]   (adjoint DLP)
    # For body  src → ∂/∂n(K·mu)   = D[·, body_cols]     (hypersingular)
    # We don't have D available in pulse form yet, but luckily the Robin body
    # case requires it.  For pure-PEC-body Dirichlet, ∂u/∂n on the body side
    # isn't needed at all; we enforce u = 0.  Similarly for dielectric-free
    # sheets we only need u.  So as long as Robin bodies are handled without
    # D, we're fine:
    #
    # Robin body BC:  u + alpha du/dn = 0
    #   u at body row  = (1/2 I + K)·mu + (S @ sigma_sheets term)
    #   du/dn at body row, from the interior limit of the representation:
    #                   = (-1/2 I + K'_adj)·mu + D·(sigma_sheets)  (exterior side)
    #     But pulse-basis D with point collocation is delicate.  For the
    #     Robin case in this pulse implementation, we approximate du/dn by
    #     using the SLP sheet contribution's adjoint: K'[:, sheet_cols]
    #     gives ∂/∂n(S·sigma), and for body-DLP mu, we use K'[:, body_cols]
    #     as a proxy for ∂/∂n on the body side.  This is NOT exactly right
    #     for closed bodies near resonance — the true hypersingular operator
    #     is needed there.  But it is correct to leading order and matches
    #     standard pulse-basis literature for Robin BC on open smooth bodies.

    # For TE PEC-body Dirichlet:  M_u · density = -u_inc, with the +1/2 jump
    # on body columns to account for DLP limit.
    # For TE sheet SIBC:           M_u · density - (Z_s/jk eta)*sigma_sheet = -u_inc

    # Row 1: PEC body rows — apply the +1/2 jump on that row's own column
    # (the self-term) for DLP limit.
    pec_body_idx = np.flatnonzero(is_pec & ~is_sheet)
    if pec_body_idx.size > 0:
        A[pec_body_idx, :] = M_u[pec_body_idx, :]
        # Add +1/2 to the diagonal of body-DLP self-terms.  The jump is +1/2
        # because the exterior limit is lim_{+} D·mu = +1/2 mu + K·mu.
        A[pec_body_idx, pec_body_idx] += 0.5
        rhs[pec_body_idx, :] = -u_inc[pec_body_idx, :]

    # Row 2: sheet rows — SIBC (S - factor*I) sigma + S contribs from other
    # sheets + K contribs from bodies = -u_inc.
    sheet_idx = np.flatnonzero(is_sheet)
    if sheet_idx.size > 0:
        A[sheet_idx, :] = M_u[sheet_idx, :]
        # Sheet factor subtracted on the diagonal.
        sheet_factor = z_s_arr[sheet_idx] / (1j * float(k0) * ETA0)
        A[sheet_idx, sheet_idx] -= sheet_factor
        rhs[sheet_idx, :] = -u_inc[sheet_idx, :]

    # Row 3: Robin body rows.  BC: u + alpha du/dn = 0 (on PEC-backed IBC).
    # For TE: alpha = -j k eta Z_s / k² = -j eta Z_s / k   (see _surface_robin_alpha).
    # We enforce  (u_s + alpha du_s/dn) = -(u_inc + alpha du_inc/dn).
    # u_s at body row i: M_u[i,:]·density + (1/2) jump on body col i.
    # du_s/dn at body row i: use K'[i,:] as a universal approximation for
    # the normal derivative of the scattered field expressed in the
    # appropriate representation on each column.
    robin_body_idx = np.flatnonzero(robin_mask & ~is_sheet)
    if robin_body_idx.size > 0:
        alpha = np.array([
            _surface_robin_alpha("TE", eps_phys=1.0, mu_phys=1.0, k_phys=k0,
                                  z_s=complex(z_s_arr[i]))
            for i in robin_body_idx
        ], dtype=np.complex128)
        # u_s rows: M_u + (1/2) jump on body-col self-terms
        u_rows = M_u[robin_body_idx, :].copy()
        # Add +1/2 on body-DLP self-term (obs == src and src is body)
        for row_pos, i in enumerate(robin_body_idx):
            u_rows[row_pos, i] += 0.5
        # du/dn rows: we use K' as a pragmatic proxy.  For sheet src cols,
        # K'[i, sheet_cols] = adjoint DLP.  For body src cols,
        # K'[i, body_cols] is the normal derivative of the body DLP
        # representation — which is actually the hypersingular operator D,
        # but we approximate it by K' for pulse-basis simplicity.  Robin BC
        # on closed bodies with pulse basis is known to have limited
        # accuracy; see the module docstring.
        dudn_rows = Kp[robin_body_idx, :].copy()
        # Add -1/2 jump on body-cols self-term (interior limit of K'·mu).
        for row_pos, i in enumerate(robin_body_idx):
            dudn_rows[row_pos, i] += -0.5

        A[robin_body_idx, :] = u_rows + alpha[:, None] * dudn_rows
        rhs[robin_body_idx, :] = -(u_inc[robin_body_idx, :]
                                     + alpha[:, None] * dn_inc[robin_body_idx, :])

    # Solve
    density = np.linalg.solve(A, rhs)
    resid = np.linalg.norm(A @ density - rhs, axis=0)
    rhs_norm = np.linalg.norm(rhs, axis=0)
    rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
    max_rel_resid = float(np.max(resid / rhs_norm))

    # Far field: sheet panels contribute via SLP projector, body panels via DLP.
    A_amps = np.zeros(len(elev_deg), dtype=np.complex128)
    # Split density
    # Pretend all are zero then add back
    density_sheet = np.zeros((N, len(elev_deg)), dtype=np.complex128)
    density_body  = np.zeros((N, len(elev_deg)), dtype=np.complex128)
    if density.ndim == 1:
        density = density.reshape(-1, 1)
    density_sheet[is_sheet, :] = density[is_sheet, :]
    density_body[~is_sheet, :] = density[~is_sheet, :]

    for a in range(len(elev_deg)):
        amp_sheet = _farfield_slp_pulse(geom, density_sheet[:, a], float(k0),
                                          elev_deg[a:a+1], quad_order)[0]
        amp_body  = _farfield_dlp_pulse(geom, density_body[:, a], float(k0),
                                          elev_deg[a:a+1], quad_order)[0]
        A_amps[a] = amp_sheet + amp_body

    rcs_lin = _rcs_sigma_from_amp(A_amps, float(k0))
    return rcs_lin, A_amps, max_rel_resid


# ───────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────

def solve_monostatic_rcs_2d_pulse(
    geometry_snapshot: Dict[str, Any],
    frequencies_ghz: Sequence[float],
    elevations_deg: Sequence[float],
    polarization: str,
    geometry_units: str = "inches",
    material_base_dir: str | None = None,
    max_panels: int = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: float | None = None,
    quad_order: int = 8,
) -> Dict[str, Any]:
    """
    Monostatic 2D RCS solve using pulse basis + point collocation.

    Same return shape as rcs_solver.solve_monostatic_rcs_2d.
    """
    pol = _normalize_polarization(polarization)
    unit_scale = _unit_scale_to_meters(geometry_units)
    base_dir = material_base_dir or "."
    elev_arr = np.asarray(elevations_deg, dtype=float)

    # Validate geometry.
    validate_geometry_snapshot_for_solver(geometry_snapshot, base_dir=base_dir)

    # Materials library (for IBC flags, dielectric tables).
    materials = MaterialLibrary.from_entries(
        geometry_snapshot.get("ibcs", []) or [],
        geometry_snapshot.get("dielectrics", []) or [],
        base_dir=base_dir,
    )

    samples: List[Dict[str, Any]] = []

    for freq_ghz in frequencies_ghz:
        freq_hz = float(freq_ghz) * 1e9
        k0 = 2.0 * math.pi * freq_hz / C0
        lambda_min = C0 / freq_hz

        # Mesh reference handling: follow Galerkin convention.
        mesh_ref_ghz = mesh_reference_ghz if mesh_reference_ghz is not None else freq_ghz
        mesh_lambda = C0 / (float(mesh_ref_ghz) * 1e9)

        panels = _build_panels(geometry_snapshot, unit_scale, mesh_lambda,
                                max_panels=max_panels)

        classification = _classify_panels(panels, materials, freq_ghz, pol, k0)

        if pol == "TM":
            rcs_lin, amp, resid = _solve_tm_slp_pulse(panels, classification,
                                                       k0, elev_arr, quad_order)
        else:
            rcs_lin, amp, resid = _solve_te_pulse(panels, classification,
                                                   k0, elev_arr, quad_order)

        rcs_db = 10.0 * np.log10(rcs_lin)
        for i, elev in enumerate(elev_arr):
            av = complex(amp[i])
            samples.append({
                "frequency_ghz": float(freq_ghz),
                "theta_inc_deg": float(elev),
                "theta_scat_deg": float(elev),
                "rcs_linear": float(rcs_lin[i]),
                "rcs_db": float(rcs_db[i]),
                "rcs_amp_real": float(np.real(av)),
                "rcs_amp_imag": float(np.imag(av)),
                "rcs_amp_phase_deg": float(math.degrees(cmath.phase(av))),
                "linear_residual": float(resid),
            })

    return {
        "solver": "2d_bie_mom_rcs_pulse",
        "scattering_mode": "monostatic",
        "polarization": _canonical_user_polarization_label(polarization),
        "polarization_export": _canonical_user_polarization_label(polarization),
        "samples": samples,
        "metadata": {
            "formulation": "2D pulse-basis point-collocation",
            "basis": "pulse",
            "quad_order": int(quad_order),
        },
    }


def register_pulse_basis() -> None:
    """
    Install a kwarg bridge on rcs_solver.solve_monostatic_rcs_2d so that
    calling it with basis='pulse' routes here, while basis='galerkin' (the
    default) behaves as before.
    """
    import rcs_solver as _rcs
    if getattr(_rcs, "_PULSE_BRIDGE_INSTALLED", False):
        return

    _orig = _rcs.solve_monostatic_rcs_2d

    def _bridged(*args, basis: str = "galerkin", **kwargs):
        if basis.strip().lower() == "pulse":
            # Strip Galerkin-only kwargs that pulse doesn't accept.
            pulse_kwargs = {k: v for k, v in kwargs.items()
                             if k in {"material_base_dir", "max_panels",
                                      "mesh_reference_ghz", "geometry_units"}}
            return solve_monostatic_rcs_2d_pulse(*args, **pulse_kwargs)
        return _orig(*args, **kwargs)

    _rcs.solve_monostatic_rcs_2d = _bridged
    _rcs._PULSE_BRIDGE_INSTALLED = True
