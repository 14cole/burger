#!/usr/bin/env python3
"""
3D expansion of a 2D monostatic RCS solve.

Takes a .grim file produced by `run_monostatic.py` / `run_solver.py` (which
stores sigma_2D per azimuth, at a single 0.0 elevation) and produces a 3D
.grim dataset compatible with GRIM (axes: azimuth, elevation, frequency,
polarization; rcs_power is linear 3D RCS in m^2).

Two modes:

1. "finite_length"  — axisymmetric body-of-revolution approximation.
   The 2D azimuth axis becomes the 3D elevation axis directly (the user's
   convention: "2D azimuths translate to 3D elevations"). The 3D azimuth
   axis is user-configurable; response is replicated across it (BOR). A
   length L (inches by default) applies a (2L/lambda) scale to convert
   sigma_2D [m] into sigma_3D [m^2].

2. "stl_xyz"  — distributed ground-point model on a real 3D body.
   User gives an STL file and a list of (x,y,z) ground points (inches by
   default). Each point is snapped to the nearest STL triangle; the
   triangle's outward normal defines the "up" axis of the point's local
   2D frame. A user-supplied GLOBAL_AXIS (e.g. the body axis) is projected
   into the tangent plane at each point to define the "right" axis of
   that frame, so az_2d = 0° always means "grazing along the body axis
   in this point's local 2D cut." The 2D lookup uses the signed in-plane
   angle, atan2(direction·n_hat, direction·t_hat), which reproduces the
   2D-solver convention (0° grazing right, 90° normal, 180° grazing left).
   When the surface normal is parallel to GLOBAL_AXIS (e.g. an end cap),
   the tangent is undefined and the lookup falls back to the rotationally
   symmetric form (90° − angle off normal). Contributions are summed
   incoherently across points (toggle COHERENT_SUM). Per-point shadow
   testing casts a ray from the point in the observation direction against
   the STL; if blocked, that point contributes zero. If every point is
   shadowed for a given direction, the output cell is forced to -200 dB.

Usage:
    python expand_3d.py

Requirements:
    Python 3.6.8+, numpy. No STL library needed — a minimal ASCII/binary
    STL reader is included below.
"""

import json
import math
import os
import struct
import sys
import time

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_GRIM  = "rcs_output.grim"
OUTPUT_GRIM = "rcs_output_3d.grim"

MODE = "finite_length"   # "finite_length" or "stl_xyz"

# ----- Mode 1: finite_length ---------------------------------------------------
LENGTH         = 36.0         # body length
LENGTH_UNITS   = "inches"     # "inches" or "meters"
# 3D azimuth grid (degrees). Response is replicated across this axis (BOR).
AZIMUTHS_3D    = list(range(0, 360, 5))

# ----- Mode 2: stl_xyz ---------------------------------------------------------
STL_PATH       = "body.stl"
STL_UNITS      = "inches"     # "inches" or "meters"
XYZ_POINTS     = [             # ground points on the body (any units)
    [0.0, 0.0, 0.0],
]
XYZ_UNITS      = "inches"
# Global body axis. Per point, the projection of this vector into the local
# tangent plane defines the 2D "az = 0° grazing right" direction. Pick the
# axis that matches how the 2D cross-section was cut (typically the long
# axis of the body). End-cap points whose normal is parallel to this axis
# fall back to the rotationally-symmetric lookup.
GLOBAL_AXIS    = (1.0, 0.0, 0.0)
# 3D grid (degrees). Used only in stl_xyz mode.
AZIMUTHS_3D_STL   = list(range(0, 360, 5))
ELEVATIONS_3D_STL = list(range(-90, 91, 5))

# Snap sanity: if the nearest STL triangle is farther than this from the
# input xyz (in XYZ_UNITS), warn — or abort if SNAP_ABORT_ON_DISTANCE is
# True. Catches typo'd coords, wrong body, wrong units.
SNAP_MAX_DISTANCE        = 0.25      # in XYZ_UNITS
SNAP_ABORT_ON_DISTANCE   = False
# Print a per-point diagnostic table after the snap + tangent-frame setup
# (input xyz, foot, distance, triangle idx, normal, tangent). Recommended
# on until you're confident the geometry is wired correctly.
SNAP_VERBOSE             = True

CHECK_SHADOWING = True
SHADOW_DB_FLOOR = -200.0      # dB floor applied when every point is shadowed
COHERENT_SUM    = False       # False = power sum; True = amplitude sum with
                              # free-space phase per point path length
# ═══════════════════════════════════════════════════════════════════════════════


C0           = 299_792_458.0
INCH_TO_M    = 0.0254
DEG2RAD      = math.pi / 180.0
SHADOW_EPS   = 1e-6           # Möller–Trumbore epsilon
RAY_OFFSET   = 1e-6           # nudge along normal to avoid self-intersection


# ─────────────────────────────────────────────────────────────────────────────
# I/O — GRIM load / save
# ─────────────────────────────────────────────────────────────────────────────

def _load_2d_grim(path):
    with open(path, "rb") as f:
        data = np.load(f, allow_pickle=False)
        out = {
            "azimuths":    np.asarray(data["azimuths"], dtype=float),
            "elevations":  np.asarray(data["elevations"], dtype=float),
            "frequencies": np.asarray(data["frequencies"], dtype=float),
            "polarizations": np.asarray(data["polarizations"], dtype=str),
            "rcs_power":   np.asarray(data["rcs_power"], dtype=np.float32),
            "rcs_phase":   np.asarray(data["rcs_phase"], dtype=np.float32),
            "units":       {},
            "history":     "",
            "source_path": "",
        }
        if "units" in data:
            try:
                out["units"] = json.loads(str(data["units"].item() or ""))
            except (ValueError, AttributeError):
                pass
        if "history" in data:
            out["history"] = str(data["history"].item() or "")
        if "source_path" in data:
            out["source_path"] = str(data["source_path"].item() or "")
    if out["elevations"].size != 1 or abs(float(out["elevations"][0])) > 1e-9:
        print("  warning: 2D input elevation is not [0.0] — expansion assumes 0.0", flush=True)
    return out


def _save_3d_grim(path, az, el, freqs, pols, rcs_power_3d, rcs_phase_3d,
                  source_path="", history=""):
    if not path.lower().endswith(".grim"):
        path = path + ".grim"
    units_payload = {
        "azimuth": "deg",
        "elevation": "deg",
        "frequency": "GHz",
        "rcs_log_unit": "dBsm",
        "rcs_linear_quantity": "sigma_3d",
    }
    with open(path, "wb") as f:
        np.savez(
            f,
            azimuths=np.asarray(az, dtype=float),
            elevations=np.asarray(el, dtype=float),
            frequencies=np.asarray(freqs, dtype=float),
            polarizations=np.asarray(pols, dtype=str),
            rcs_power=rcs_power_3d.astype(np.float32),
            rcs_phase=rcs_phase_3d.astype(np.float32),
            rcs_domain="power_phase",
            power_domain="linear_rcs",
            source_path=source_path,
            history=history,
            units=json.dumps(units_payload),
        )
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Physics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _length_to_meters(value, unit):
    u = str(unit).strip().lower()
    if u in ("m", "meter", "meters"):
        return float(value)
    if u in ("in", "inch", "inches"):
        return float(value) * INCH_TO_M
    raise ValueError("unknown length unit: {}".format(unit))


def _wavelengths_m(freqs_ghz):
    freqs_hz = np.asarray(freqs_ghz, dtype=float) * 1.0e9
    return C0 / freqs_hz


def _dir_from_az_el(az_deg, el_deg):
    """Unit vector pointing *toward* observer at (az, el).

    Compass / radar NED convention:
      - azimuth measured clockwise in the xy-plane from +y
        (0° = +y, 90° = +x, 180° = -y, 270° = -x, viewed from +z)
      - elevation measured from the xy-plane toward +z
        (0° = horizon, +90° = +z, -90° = -z)
    """
    ca = math.cos(az_deg * DEG2RAD)
    sa = math.sin(az_deg * DEG2RAD)
    ce = math.cos(el_deg * DEG2RAD)
    se = math.sin(el_deg * DEG2RAD)
    return np.array([sa * ce, ca * ce, se], dtype=float)


def _angle_between(v1, v2):
    """Angle between unit vectors, robust in [0, pi]."""
    c = float(np.dot(v1, v2))
    c = max(-1.0, min(1.0, c))
    return math.acos(c)


# ─────────────────────────────────────────────────────────────────────────────
# Minimal STL reader (binary + ASCII), returns (tris Nx3x3, normals Nx3)
# ─────────────────────────────────────────────────────────────────────────────

def _read_stl(path):
    with open(path, "rb") as f:
        head = f.read(5)
        f.seek(0)
        if head == b"solid":
            # Could still be binary (non-conforming). Try ASCII first.
            try:
                return _read_stl_ascii(path)
            except Exception:
                return _read_stl_binary(path)
        return _read_stl_binary(path)


def _read_stl_binary(path):
    with open(path, "rb") as f:
        f.read(80)  # header
        (n_tri,) = struct.unpack("<I", f.read(4))
        tris = np.empty((n_tri, 3, 3), dtype=np.float64)
        normals = np.empty((n_tri, 3), dtype=np.float64)
        for i in range(n_tri):
            rec = f.read(50)
            vals = struct.unpack("<12fH", rec)
            normals[i] = vals[0:3]
            tris[i, 0] = vals[3:6]
            tris[i, 1] = vals[6:9]
            tris[i, 2] = vals[9:12]
    # Recompute normals (many STL writers leave zeros).
    normals = _recompute_normals(tris, provided=normals)
    return tris, normals


def _read_stl_ascii(path):
    tris = []
    normals = []
    cur_n = None
    cur_verts = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "facet" and len(parts) >= 5 and parts[1] == "normal":
                cur_n = [float(parts[2]), float(parts[3]), float(parts[4])]
                cur_verts = []
            elif parts[0] == "vertex" and len(parts) >= 4:
                cur_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "endfacet":
                if len(cur_verts) == 3 and cur_n is not None:
                    tris.append(cur_verts)
                    normals.append(cur_n)
                cur_n = None
                cur_verts = []
    tris_arr = np.asarray(tris, dtype=np.float64)
    normals_arr = np.asarray(normals, dtype=np.float64)
    normals_arr = _recompute_normals(tris_arr, provided=normals_arr)
    return tris_arr, normals_arr


def _recompute_normals(tris, provided=None):
    e1 = tris[:, 1, :] - tris[:, 0, :]
    e2 = tris[:, 2, :] - tris[:, 0, :]
    n = np.cross(e1, e2)
    mag = np.linalg.norm(n, axis=1)
    safe = mag > 0.0
    out = np.zeros_like(n)
    out[safe] = n[safe] / mag[safe, None]
    # If provided normals are plausible, prefer their orientation sign.
    if provided is not None and provided.shape == out.shape:
        provided_mag = np.linalg.norm(provided, axis=1)
        has_provided = provided_mag > 1e-6
        if np.any(has_provided):
            dots = np.einsum("ij,ij->i", out[has_provided], provided[has_provided])
            flip = dots < 0.0
            idxs = np.where(has_provided)[0][flip]
            out[idxs] = -out[idxs]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Nearest-triangle snap + Möller–Trumbore ray intersection
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_triangle(point, tris):
    """Return (tri_index, foot_point, distance) — brute-force O(N), OK for
    typical STL sizes up to ~1e5 triangles."""
    best_i = 0
    best_d = math.inf
    best_p = np.zeros(3)
    for i in range(tris.shape[0]):
        p = _closest_point_on_triangle(point, tris[i, 0], tris[i, 1], tris[i, 2])
        d = float(np.linalg.norm(p - point))
        if d < best_d:
            best_d = d
            best_i = i
            best_p = p
    return best_i, best_p, best_d


def _closest_point_on_triangle(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy()
    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b.copy()
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3) if (d1 - d3) != 0.0 else 0.0
        return a + v * ab
    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c.copy()
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6) if (d2 - d6) != 0.0 else 0.0
        return a + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / denom if denom != 0.0 else 0.0
        return b + w * (c - b)
    denom = va + vb + vc
    if denom <= 0.0:
        return a.copy()
    v = vb / denom
    w = vc / denom
    return a + ab * v + ac * w


def _mt_any_hit(origin, direction, tris_subset, orig_indices, skip_idx):
    """Vectorized Möller–Trumbore any-hit over a subset of triangles.

    `orig_indices[k]` is the original index of the k-th triangle in the
    subset; used only to honor `skip_idx`.
    """
    if tris_subset.shape[0] == 0:
        return False
    v0 = tris_subset[:, 0, :]
    v1 = tris_subset[:, 1, :]
    v2 = tris_subset[:, 2, :]
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(direction, e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > SHADOW_EPS
    s = origin - v0
    u = np.zeros_like(a)
    u[ok] = np.einsum("ij,ij->i", s[ok], h[ok]) / a[ok]
    ok &= (u >= 0.0) & (u <= 1.0)
    q = np.cross(s, e1)
    v = np.zeros_like(a)
    v[ok] = np.einsum("j,ij->i", direction, q[ok]) / a[ok]
    ok &= (v >= 0.0) & (u + v <= 1.0)
    t = np.full_like(a, np.inf)
    t[ok] = np.einsum("ij,ij->i", e2[ok], q[ok]) / a[ok]
    ok &= (t > SHADOW_EPS)
    if skip_idx >= 0:
        ok &= (orig_indices != skip_idx)
    return bool(np.any(ok))


def _ray_hits_any_triangle(origin, direction, tris, skip_idx=-1):
    """Brute-force Möller–Trumbore over all triangles. Retained as a
    correctness reference and for tiny meshes; use the BVH query for real
    workloads."""
    orig = np.arange(tris.shape[0], dtype=np.int64)
    return _mt_any_hit(origin, direction, tris, orig, skip_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Minimal BVH (median-split, flat-array storage) for fast shadow queries
# ─────────────────────────────────────────────────────────────────────────────
#
# Node encoding:
#   bmin[i], bmax[i]  : (3,) AABB of node i
#   left[i]           : >= 0 → child index (internal node)
#                       <  0 → leaf; start = -left[i] - 1
#   right[i]          : child index (internal) or end (leaf, exclusive)
#   tri_indices       : (n_tri,) permutation of original triangle indices;
#                       a leaf covers tri_indices[start:end]

def _build_bvh(tris, leaf_size=8):
    n_tri = int(tris.shape[0])
    if n_tri == 0:
        return None
    centroids = tris.mean(axis=1).astype(np.float64)
    tri_indices = np.arange(n_tri, dtype=np.int64)

    nbm, nbx, nlf, nrt = [], [], [], []

    def _build(start, end):
        node_idx = len(nbm)
        nbm.append(None); nbx.append(None); nlf.append(None); nrt.append(None)

        sel = tri_indices[start:end]
        verts = tris[sel].reshape(-1, 3)
        bmin = verts.min(axis=0)
        bmax = verts.max(axis=0)

        if end - start <= leaf_size:
            nbm[node_idx] = bmin
            nbx[node_idx] = bmax
            nlf[node_idx] = -(start + 1)
            nrt[node_idx] = end
            return node_idx

        axis = int(np.argmax(bmax - bmin))
        order = np.argsort(centroids[sel, axis])
        tri_indices[start:end] = sel[order]
        mid = start + (end - start) // 2

        left = _build(start, mid)
        right = _build(mid, end)

        nbm[node_idx] = bmin
        nbx[node_idx] = bmax
        nlf[node_idx] = left
        nrt[node_idx] = right
        return node_idx

    _build(0, n_tri)
    return {
        "bmin": np.asarray(nbm, dtype=np.float64),
        "bmax": np.asarray(nbx, dtype=np.float64),
        "left": np.asarray(nlf, dtype=np.int64),
        "right": np.asarray(nrt, dtype=np.int64),
        "tri_indices": tri_indices,
    }


def _ray_aabb_slabs(origin, inv_dir, bmin, bmax):
    """Standard slabs test. Returns (tmin, tmax); no hit when tmax < max(0, tmin)."""
    t0 = (bmin - origin) * inv_dir
    t1 = (bmax - origin) * inv_dir
    tlo = np.minimum(t0, t1)
    thi = np.maximum(t0, t1)
    return float(tlo.max()), float(thi.min())


def _ray_hits_any_bvh(origin, direction, tris, bvh, skip_idx=-1):
    """Iterative stack-based any-hit ray query."""
    if bvh is None:
        return False
    # Guard against zero direction components.
    dir_safe = np.where(np.abs(direction) < 1e-300, 1e-300, direction)
    inv_dir = 1.0 / dir_safe

    bmin = bvh["bmin"]
    bmax = bvh["bmax"]
    left = bvh["left"]
    right = bvh["right"]
    tri_indices = bvh["tri_indices"]

    stack = [0]
    while stack:
        node = stack.pop()
        tmin, tmax = _ray_aabb_slabs(origin, inv_dir, bmin[node], bmax[node])
        if tmax < max(0.0, tmin) or tmax < SHADOW_EPS:
            continue
        l = int(left[node])
        if l < 0:
            start = -l - 1
            end = int(right[node])
            orig = tri_indices[start:end]
            if _mt_any_hit(origin, direction, tris[orig], orig, skip_idx):
                return True
        else:
            # Push far-side second (popped first = visit near-side first).
            stack.append(int(right[node]))
            stack.append(l)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: finite_length
# ─────────────────────────────────────────────────────────────────────────────

def _expand_finite_length(data_2d):
    az_2d = data_2d["azimuths"]
    freqs = data_2d["frequencies"]
    pols = data_2d["polarizations"]
    power_2d = data_2d["rcs_power"]    # (n_az_2d, 1, n_f, n_pol) — sigma_2D linear
    phase_2d = data_2d["rcs_phase"]

    # 2D az axis becomes 3D elevation axis.
    elevations_3d = az_2d.astype(float)
    azimuths_3d = np.asarray(AZIMUTHS_3D, dtype=float)

    L_m = _length_to_meters(LENGTH, LENGTH_UNITS)
    lambdas = _wavelengths_m(freqs)           # (n_f,)
    # sigma_3D = sigma_2D * (2 L / lambda). Per-freq scalar, broadcast across el.
    length_factor = (2.0 * L_m) / lambdas      # (n_f,)

    n_az = len(azimuths_3d)
    n_el = len(elevations_3d)
    n_f  = len(freqs)
    n_p  = len(pols)

    # power_2d shape: (n_az_2d, 1, n_f, n_pol). Drop elev dim, reorder for broadcast.
    # After squeeze: (n_el_3d, n_f, n_pol). Replicate across new az axis.
    base_power = np.squeeze(power_2d, axis=1)     # (n_el_3d, n_f, n_pol)
    base_phase = np.squeeze(phase_2d, axis=1)
    scaled = base_power * length_factor[None, :, None]   # apply (2L/lambda)

    out_power = np.broadcast_to(scaled[None, :, :, :],
                                (n_az, n_el, n_f, n_p)).astype(np.float32).copy()
    out_phase = np.broadcast_to(base_phase[None, :, :, :],
                                (n_az, n_el, n_f, n_p)).astype(np.float32).copy()
    return azimuths_3d, elevations_3d, freqs, pols, out_power, out_phase


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: stl_xyz
# ─────────────────────────────────────────────────────────────────────────────

def _expand_stl_xyz(data_2d):
    az_2d = data_2d["azimuths"].astype(float)
    freqs = data_2d["frequencies"].astype(float)
    pols = data_2d["polarizations"]
    power_2d = np.squeeze(data_2d["rcs_power"], axis=1)   # (n_az_2d, n_f, n_pol)
    phase_2d = np.squeeze(data_2d["rcs_phase"], axis=1)

    tris_raw, tri_normals = _read_stl(STL_PATH)
    stl_scale = _length_to_meters(1.0, STL_UNITS)
    xyz_scale = _length_to_meters(1.0, XYZ_UNITS)
    tris = tris_raw * stl_scale
    pts = np.asarray(XYZ_POINTS, dtype=float) * xyz_scale

    # Snap each point to the nearest triangle, pick up its normal.
    point_tri = np.empty(pts.shape[0], dtype=np.int64)
    point_feet = np.empty_like(pts)
    point_normals = np.empty_like(pts)
    point_snap_dist = np.empty(pts.shape[0], dtype=np.float64)
    for pi in range(pts.shape[0]):
        ti, foot, dist = _nearest_triangle(pts[pi], tris)
        point_tri[pi] = ti
        point_feet[pi] = foot
        point_normals[pi] = tri_normals[ti]
        point_snap_dist[pi] = dist

    # Snap-distance guard. Threshold is expressed in XYZ_UNITS; convert to m
    # to compare against the internal (already-scaled) distance.
    snap_threshold_m = _length_to_meters(float(SNAP_MAX_DISTANCE), XYZ_UNITS)
    far_points = np.where(point_snap_dist > snap_threshold_m)[0]
    if far_points.size > 0:
        msg_lines = []
        for pi in far_points:
            dist_user = point_snap_dist[pi] / xyz_scale
            in_xyz = np.asarray(XYZ_POINTS[pi], dtype=float)
            msg_lines.append(
                "    point {}: input={} {}  snap_dist={:.4g} {} "
                "(threshold {:.4g})".format(
                    pi, list(in_xyz), XYZ_UNITS,
                    dist_user, XYZ_UNITS, float(SNAP_MAX_DISTANCE),
                )
            )
        banner = (
            "{} point(s) snapped farther than SNAP_MAX_DISTANCE "
            "— check coords, units, and that STL is the correct body:"
        ).format(far_points.size)
        if SNAP_ABORT_ON_DISTANCE:
            raise RuntimeError(banner + "\n" + "\n".join(msg_lines))
        print("  WARNING: " + banner, flush=True)
        for line in msg_lines:
            print(line, flush=True)

    # Build per-point tangent frames. The global body axis, projected into
    # each point's tangent plane, becomes that point's 2D "az = 0° grazing
    # right" direction. If the axis is parallel to the normal, the tangent
    # is undefined and we flag the point for the rotationally-symmetric
    # fallback.
    axis = np.asarray(GLOBAL_AXIS, dtype=float)
    axis_mag = float(np.linalg.norm(axis))
    if axis_mag < 1e-9:
        print("  warning: GLOBAL_AXIS is zero — defaulting to (1, 0, 0)", flush=True)
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        axis = axis / axis_mag
    point_tangents = np.zeros_like(pts)
    point_tangent_valid = np.zeros(pts.shape[0], dtype=bool)
    for pi in range(pts.shape[0]):
        n_hat = point_normals[pi]
        t_raw = axis - float(np.dot(axis, n_hat)) * n_hat
        t_mag = float(np.linalg.norm(t_raw))
        if t_mag > 1e-6:
            point_tangents[pi] = t_raw / t_mag
            point_tangent_valid[pi] = True
        else:
            print(
                "  warning: point {} normal is parallel to GLOBAL_AXIS — "
                "falling back to rotationally-symmetric lookup".format(pi),
                flush=True,
            )

    if SNAP_VERBOSE:
        print("  per-point snap diagnostic (distances in {}):".format(XYZ_UNITS), flush=True)
        print(
            "    {:>3}  {:>20}  {:>20}  {:>9}  {:>6}  {:>20}  {:>20}  {:>5}".format(
                "#", "input", "foot", "dist", "tri", "normal", "tangent", "t_ok"
            ),
            flush=True,
        )
        for pi in range(pts.shape[0]):
            inp = np.asarray(XYZ_POINTS[pi], dtype=float)
            foot_user = point_feet[pi] / xyz_scale
            dist_user = point_snap_dist[pi] / xyz_scale
            nvec = point_normals[pi]
            tvec = point_tangents[pi]
            print(
                "    {:>3d}  ({: .3f},{: .3f},{: .3f})  "
                "({: .3f},{: .3f},{: .3f})  {:>9.4g}  {:>6d}  "
                "({: .3f},{: .3f},{: .3f})  ({: .3f},{: .3f},{: .3f})  {:>5}".format(
                    pi, inp[0], inp[1], inp[2],
                    foot_user[0], foot_user[1], foot_user[2],
                    dist_user, int(point_tri[pi]),
                    nvec[0], nvec[1], nvec[2],
                    tvec[0], tvec[1], tvec[2],
                    "yes" if point_tangent_valid[pi] else "NO",
                ),
                flush=True,
            )

    # Build a BVH once for the whole shadow-ray workload.
    bvh = None
    if CHECK_SHADOWING:
        t_bvh = time.time()
        bvh = _build_bvh(tris, leaf_size=8)
        print("  BVH: {} nodes for {} tris  ({:.2f}s)".format(
            0 if bvh is None else bvh["bmin"].shape[0],
            tris.shape[0], time.time() - t_bvh), flush=True)

    azimuths_3d = np.asarray(AZIMUTHS_3D_STL, dtype=float)
    elevations_3d = np.asarray(ELEVATIONS_3D_STL, dtype=float)
    n_az, n_el, n_f, n_p = len(azimuths_3d), len(elevations_3d), len(freqs), len(pols)

    out_power = np.zeros((n_az, n_el, n_f, n_p), dtype=np.float64)
    out_phase = np.full_like(out_power, np.nan, dtype=np.float64)

    # 2D sigma lookup: snap angle to the nearest az_2d bin.
    # The 2D solver's azimuth axis is assumed to span the full needed range
    # (typically 0..360 or -180..180). Interpolation done linearly in angle.
    az_2d_sorted_idx = np.argsort(az_2d)
    az_2d_sorted = az_2d[az_2d_sorted_idx]
    power_2d_sorted = power_2d[az_2d_sorted_idx]
    phase_2d_sorted = phase_2d[az_2d_sorted_idx]

    def lookup_2d(angle_deg):
        # Wrap to the 2D axis range.
        lo, hi = float(az_2d_sorted[0]), float(az_2d_sorted[-1])
        span = hi - lo
        if span <= 0.0:
            return power_2d_sorted[0], phase_2d_sorted[0]
        a = angle_deg
        # Try wrapping by 360 if needed.
        while a < lo:
            a += 360.0
        while a > hi:
            a -= 360.0
        if a < lo or a > hi:
            a = max(lo, min(hi, angle_deg))
        # Linear interp in magnitude; take nearest-neighbor phase to avoid
        # unwrap artifacts from float32 storage.
        ju = np.searchsorted(az_2d_sorted, a)
        if ju <= 0:
            return power_2d_sorted[0], phase_2d_sorted[0]
        if ju >= len(az_2d_sorted):
            return power_2d_sorted[-1], phase_2d_sorted[-1]
        jl = ju - 1
        t = (a - az_2d_sorted[jl]) / (az_2d_sorted[ju] - az_2d_sorted[jl])
        pw = (1.0 - t) * power_2d_sorted[jl] + t * power_2d_sorted[ju]
        ph = phase_2d_sorted[jl] if t < 0.5 else phase_2d_sorted[ju]
        return pw, ph

    lambdas = _wavelengths_m(freqs)           # (n_f,)
    k0 = 2.0 * math.pi / lambdas              # (n_f,)

    n_points = pts.shape[0]
    print("  ground points: {}   tris: {}".format(n_points, tris.shape[0]), flush=True)

    t_start = time.time()
    for ia in range(n_az):
        if ia % max(1, n_az // 10) == 0 and ia > 0:
            dt = time.time() - t_start
            print("  progress: az {}/{}  ({:.1f}s)".format(ia, n_az, dt), flush=True)
        for ie in range(n_el):
            direction = _dir_from_az_el(azimuths_3d[ia], elevations_3d[ie])
            # Per-point: local angle vs normal and shadow test (direction-indep
            # shadow only — actually does depend on direction; computed below).
            lit_any = False
            # Per-freq accumulator: coherent amplitude (complex) or power.
            if COHERENT_SUM:
                amp_acc = np.zeros((n_f, n_p), dtype=np.complex128)
            else:
                pow_acc = np.zeros((n_f, n_p), dtype=np.float64)
            for ip in range(n_points):
                n_hat = point_normals[ip]
                # Only front faces contribute (normal must face observer).
                if float(np.dot(n_hat, direction)) <= 0.0:
                    continue
                if CHECK_SHADOWING:
                    origin = point_feet[ip] + RAY_OFFSET * n_hat
                    if _ray_hits_any_bvh(origin, direction, tris, bvh,
                                         skip_idx=int(point_tri[ip])):
                        continue
                lit_any = True
                # 2D axis convention: az_2d = 0° grazing right (along the
                # global body axis projected into this point's tangent plane),
                # 90° normal, 180° grazing left. Use the signed in-plane
                # angle so asymmetric 2D cuts map correctly. On points whose
                # normal is parallel to GLOBAL_AXIS, fall back to the
                # rotationally-symmetric form.
                if point_tangent_valid[ip]:
                    t_hat = point_tangents[ip]
                    dn = float(np.dot(n_hat, direction))
                    dt = float(np.dot(t_hat, direction))
                    az_2d_lookup_deg = math.degrees(math.atan2(dn, dt))
                else:
                    az_2d_lookup_deg = 90.0 - math.degrees(_angle_between(n_hat, direction))
                pw, ph = lookup_2d(az_2d_lookup_deg)
                # pw, ph shape: (n_f, n_pol)
                if COHERENT_SUM:
                    # Path-length phase: exp(-j*2k*r) for monostatic round trip
                    # referenced to foot point along observation direction.
                    r = float(np.dot(point_feet[ip], direction))
                    amp = np.sqrt(np.maximum(pw, 0.0)) \
                          * np.exp(1j * ph) \
                          * np.exp(-1j * 2.0 * k0[:, None] * r)
                    amp_acc += amp
                else:
                    pow_acc += np.maximum(pw, 0.0)
            if COHERENT_SUM:
                out_power[ia, ie] = np.abs(amp_acc) ** 2
                out_phase[ia, ie] = np.angle(amp_acc)
            else:
                out_power[ia, ie] = pow_acc
                out_phase[ia, ie] = np.nan
            if not lit_any:
                # All points shadowed / back-facing — write the dB floor.
                out_power[ia, ie] = 10.0 ** (SHADOW_DB_FLOOR / 10.0)
                out_phase[ia, ie] = np.nan
    return azimuths_3d, elevations_3d, freqs, pols, \
           out_power.astype(np.float32), out_phase.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isfile(INPUT_GRIM):
        print("ERROR: input not found: {}".format(INPUT_GRIM))
        sys.exit(1)

    print("=" * 70)
    print("3D expansion of 2D monostatic RCS   (mode: {})".format(MODE))
    print("=" * 70)
    data_2d = _load_2d_grim(INPUT_GRIM)
    print("  input    : {}".format(INPUT_GRIM))
    print("  2D az    : {} points, {:.4g} to {:.4g} deg".format(
        data_2d["azimuths"].size,
        float(data_2d["azimuths"].min()), float(data_2d["azimuths"].max())))
    print("  freq     : {} points, {:.4g} to {:.4g} GHz".format(
        data_2d["frequencies"].size,
        float(data_2d["frequencies"].min()), float(data_2d["frequencies"].max())))
    print("  pol      : {}".format(", ".join(str(p) for p in data_2d["polarizations"])))
    print()

    t0 = time.time()
    if MODE == "finite_length":
        L_m = _length_to_meters(LENGTH, LENGTH_UNITS)
        print("  length   : {} {}  = {:.6g} m".format(LENGTH, LENGTH_UNITS, L_m))
        print("  3D az    : {} points".format(len(AZIMUTHS_3D)))
        az, el, fr, pol, power, phase = _expand_finite_length(data_2d)
    elif MODE == "stl_xyz":
        print("  STL      : {}   units: {}".format(STL_PATH, STL_UNITS))
        print("  XYZ pts  : {}   units: {}".format(len(XYZ_POINTS), XYZ_UNITS))
        print("  3D az    : {} points".format(len(AZIMUTHS_3D_STL)))
        print("  3D el    : {} points".format(len(ELEVATIONS_3D_STL)))
        print("  shadowing: {}   summation: {}".format(
            CHECK_SHADOWING, "coherent" if COHERENT_SUM else "power"))
        az, el, fr, pol, power, phase = _expand_stl_xyz(data_2d)
    else:
        print("ERROR: unknown MODE {!r}".format(MODE))
        sys.exit(1)

    dt = time.time() - t0
    print()
    print("  expanded to shape {}  in {:.2f} s".format(power.shape, dt))

    # Reporting range (ignore non-finite).
    finite = np.isfinite(power) & (power > 0)
    if np.any(finite):
        db = 10.0 * np.log10(power[finite])
        print("  RCS range: {:.2f} to {:.2f} dBsm".format(float(db.min()), float(db.max())))

    out = _save_3d_grim(
        OUTPUT_GRIM, az, el, fr, pol, power, phase,
        source_path=os.path.abspath(INPUT_GRIM),
        history="expand_3d.py mode={}".format(MODE),
    )
    print("  wrote    : {}".format(out))
    print("Done.")


if __name__ == "__main__":
    main()
