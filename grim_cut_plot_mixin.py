from __future__ import annotations

import numpy as np

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QListWidget,
    QToolButton,
)

from grim_dataset import RcsGrid
from plot_modes import (
    azimuth_polar_mode,
    azimuth_rect_mode,
    compare_mode,
    elevation_sweep_mode,
    frequency_mode,
    isar_3d_mode,
    isar_mode,
    waterfall_mode,
)


class PlotOpsMixin:
    def _on_param_selection_changed(self) -> None:
        self._maybe_autoplot()

    def _on_polarization_selection_changed(self) -> None:
        if self.active_dataset is None:
            return
        selected_pol = sorted(self._selected_indices(self.list_pol))
        if not selected_pol:
            self._fill_list(self.list_freq, self.active_dataset.frequencies)
            self._fill_list(self.list_elev, self.active_dataset.elevations)
            self._fill_list(self.list_az, self.active_dataset.azimuths)
            return

        prev_freq = self._selected_indices(self.list_freq)
        prev_elev = self._selected_indices(self.list_elev)
        prev_az = self._selected_indices(self.list_az)

        pwr_sel = self.active_dataset.rcs_power[:, :, :, selected_pol]
        if self._button_checked(self.btn_phase):
            phs_sel = self.active_dataset.rcs_phase[:, :, :, selected_pol]
            valid = np.isfinite(pwr_sel) & np.isfinite(phs_sel)
        else:
            valid = np.isfinite(pwr_sel)
        freq_avail = valid.any(axis=(0, 1, 3))
        elev_avail = valid.any(axis=(0, 2, 3))
        az_avail = valid.any(axis=(1, 2, 3))

        freq_indices = np.where(freq_avail)[0]
        elev_indices = np.where(elev_avail)[0]
        az_indices = np.where(az_avail)[0]

        self._fill_list(self.list_freq, self.active_dataset.frequencies, freq_indices)
        self._fill_list(self.list_elev, self.active_dataset.elevations, elev_indices)
        self._fill_list(self.list_az, self.active_dataset.azimuths, az_indices)

        self._reselect_indices(self.list_freq, prev_freq)
        self._reselect_indices(self.list_elev, prev_elev)
        self._reselect_indices(self.list_az, prev_az)
        self._maybe_autoplot()

    def _maybe_autoplot(self) -> None:
        if not self._button_checked(self.btn_auto_plot):
            return
        if self.last_plot_mode is None:
            return
        if self.last_plot_mode == "azimuth_rect":
            self._plot_azimuth_rect()
        elif self.last_plot_mode == "azimuth_polar":
            self._plot_azimuth_polar()
        elif self.last_plot_mode == "frequency":
            self._plot_frequency()
        elif self.last_plot_mode == "elevation_sweep":
            self._plot_elevation_sweep()
        elif self.last_plot_mode == "waterfall":
            self._plot_waterfall()
        elif self.last_plot_mode == "isar_image":
            self._plot_isar_image()
        elif self.last_plot_mode == "isar_3d":
            self._plot_isar_3d()
        elif self.last_plot_mode == "compare":
            self._plot_compare()

    def _on_pbp_toggled(self) -> None:
        if self.last_plot_mode is None:
            return
        if self.last_plot_mode == "azimuth_rect":
            self._plot_azimuth_rect()
        elif self.last_plot_mode == "azimuth_polar":
            self._plot_azimuth_polar()
        elif self.last_plot_mode == "frequency":
            self._plot_frequency()
        elif self.last_plot_mode == "isar_image":
            self._plot_isar_image()

    def _on_waterfall_style_changed(self) -> None:
        if self.last_plot_mode not in ("waterfall", "isar_image", "isar_3d"):
            return
        if self.last_plot_mode == "waterfall":
            self._plot_waterfall()
        elif self.last_plot_mode == "isar_3d":
            self._plot_isar_3d()
        else:
            self._plot_isar_image()

    def _on_colormap_changed(self) -> None:
        if self.last_plot_mode == "waterfall":
            self._plot_waterfall()
            return
        if self.last_plot_mode == "isar_image":
            self._plot_isar_image()
            return
        if self.last_plot_mode == "isar_3d":
            self._plot_isar_3d()
            return
        if self.pbp_fill_mode not in ("heatmap_rcs", "heatmap_density"):
            return
        if self.last_plot_mode == "azimuth_rect":
            self._plot_azimuth_rect()
        elif self.last_plot_mode == "azimuth_polar":
            self._plot_azimuth_polar()
        elif self.last_plot_mode == "frequency":
            self._plot_frequency()

    def _update_isar3d_thin_controls(self) -> None:
        enabled = bool(self.chk_isar3d_auto_thin.isChecked())
        self.spin_isar3d_max_az.setEnabled(enabled)
        self.spin_isar3d_max_el.setEnabled(enabled)
        self.spin_isar3d_max_freq.setEnabled(enabled)

    def _on_isar3d_auto_thin_toggled(self) -> None:
        self._update_isar3d_thin_controls()
        if self.last_plot_mode == "isar_3d":
            self._plot_isar_3d()

    def _on_isar_3d_style_changed(self) -> None:
        if self.last_plot_mode == "isar_3d":
            self._plot_isar_3d()

    def _on_plot_scale_changed(self) -> None:
        if self.last_plot_mode is None:
            self._apply_plot_theme()
            return
        if self.last_plot_mode == "azimuth_rect":
            self._plot_azimuth_rect()
            self._fit_y()
        elif self.last_plot_mode == "azimuth_polar":
            self._plot_azimuth_polar()
            self._fit_y()
        elif self.last_plot_mode == "frequency":
            self._plot_frequency()
            self._fit_y()
        elif self.last_plot_mode == "elevation_sweep":
            self._plot_elevation_sweep()
            self._fit_y()
        elif self.last_plot_mode == "waterfall":
            self._plot_waterfall()
        elif self.last_plot_mode == "isar_image":
            self._plot_isar_image()
        elif self.last_plot_mode == "isar_3d":
            self._plot_isar_3d()

    def _plot_scale_mode(self) -> str:
        scale = self.combo_plot_scale.currentData()
        if scale in ("dbsm", "linear"):
            return scale
        return "dbsm"

    def _plot_scale_is_linear(self) -> bool:
        return self._plot_scale_mode() == "linear"

    def _rcs_display_values(self, dataset: RcsGrid, rcs_values, frequency_value=None):
        if self._button_checked(self.btn_phase):
            return np.degrees(np.angle(rcs_values))
        if self._plot_scale_is_linear():
            return dataset.rcs_to_linear(rcs_values)
        return dataset.rcs_to_display_db(rcs_values, frequency_value=frequency_value)

    def _rcs_axis_label(self) -> str:
        if self._button_checked(self.btn_phase):
            return "Phase (deg)"
        if self._plot_scale_is_linear():
            return "RCS (Linear)"
        return "RCS (dBsm)"

    def _rcs_p50_axis_label(self) -> str:
        if self._button_checked(self.btn_phase):
            return "Phase P50 (deg)"
        if self._plot_scale_is_linear():
            return "RCS P50 (Linear)"
        return "RCS P50 (dBsm)"

    def _polar_zero_location(self) -> str:
        loc = self.combo_polar_zero.currentData()
        if isinstance(loc, str) and loc:
            return loc
        return "N"

    def _apply_polar_orientation(self, ax) -> None:
        if ax.name != "polar":
            return
        ax.set_theta_zero_location(self._polar_zero_location())
        # Compass convention: azimuth increases clockwise.
        ax.set_theta_direction(-1)
        # Label tick marks in (-180, 180] so -90 shows on the left (W) under
        # the default N-up/CW orientation, matching the compass grid.
        tick_deg = np.arange(0, 360, 30)
        labels = [str(int(t if t <= 180 else t - 360)) for t in tick_deg]
        ax.set_thetagrids(tick_deg, labels=labels)

    def _apply_polar_zero_direction(self) -> None:
        axes = self.plot_axes or [self.plot_ax]
        for ax in axes:
            self._apply_polar_orientation(ax)

    def _on_polar_zero_changed(self) -> None:
        self._apply_polar_zero_direction()
        self.plot_canvas.draw_idle()

    def _edges_from_centers(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 1:
            step = 1.0
            return np.array([values[0] - 0.5 * step, values[0] + 0.5 * step], dtype=float)
        diffs = np.diff(values)
        edges = np.empty(values.size + 1, dtype=float)
        edges[1:-1] = values[:-1] + diffs / 2.0
        edges[0] = values[0] - diffs[0] / 2.0
        edges[-1] = values[-1] + diffs[-1] / 2.0
        return edges

    def _plot_pbp_heatmap(
        self,
        x_values,
        y_min,
        y_max,
        *,
        density: np.ndarray | None = None,
    ) -> None:
        x_values = np.asarray(x_values, dtype=float)
        y_min = np.asarray(y_min, dtype=float)
        y_max = np.asarray(y_max, dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_min) & np.isfinite(y_max)
        if not np.any(valid):
            return

        def draw_segment(seg_x, seg_min, seg_max, seg_density=None) -> None:
            lower = np.minimum(seg_min, seg_max)
            upper = np.maximum(seg_min, seg_max)
            if seg_x.size == 0:
                return

            x_edges = self._edges_from_centers(seg_x)
            lower_edges = np.interp(x_edges, seg_x, lower, left=lower[0], right=lower[-1])
            upper_edges = np.interp(x_edges, seg_x, upper, left=upper[0], right=upper[-1])

            samples = max(8, int(self.pbp_heatmap_samples))
            y_edges = np.vstack(
                [np.linspace(lo, hi, samples + 1) for lo, hi in zip(lower_edges, upper_edges)]
            ).T
            if self.pbp_fill_mode == "heatmap_density":
                if seg_density is None:
                    return
                values = np.tile(seg_density, (samples, 1))
            else:
                values = np.vstack(
                    [np.linspace(lo, hi, samples) for lo, hi in zip(lower, upper)]
                ).T
            x_grid = np.tile(x_edges, (samples + 1, 1))

            cmap = self._effective_colormap()
            self.plot_ax.pcolormesh(x_grid, y_edges, values, shading="auto", cmap=cmap)

        start = None
        for idx, is_valid in enumerate(valid):
            if is_valid and start is None:
                start = idx
            elif not is_valid and start is not None:
                seg = slice(start, idx)
                seg_density = None
                if density is not None:
                    seg_density = np.asarray(density, dtype=float)[seg]
                draw_segment(x_values[seg], y_min[seg], y_max[seg], seg_density)
                start = None
        if start is not None:
            seg = slice(start, len(valid))
            seg_density = None
            if density is not None:
                seg_density = np.asarray(density, dtype=float)[seg]
            draw_segment(x_values[seg], y_min[seg], y_max[seg], seg_density)

    def _plot_pbp_fill(
        self,
        x_values,
        y_min,
        y_max,
        label: str,
        polar: bool,
        *,
        density: np.ndarray | None = None,
    ) -> None:
        if self.pbp_fill_mode in ("heatmap_rcs", "heatmap_density"):
            self._plot_pbp_heatmap(x_values, y_min, y_max, density=density)
            self.plot_ax.plot([], [], color=self.pbp_fill_gray, label=label)
            return
        self.plot_ax.fill_between(
            x_values,
            y_min,
            y_max,
            color=self.pbp_fill_gray,
            alpha=1.0,
            label=label,
        )

    def _style_axes(self, ax) -> None:
        bg = self._current_plot_bg()
        grid = self._current_plot_grid()
        text = self._current_plot_text()
        ax.set_facecolor(bg)
        grid_on = self._plot_grid_enabled()
        ax.grid(grid_on, color=grid, alpha=0.35)
        ax.tick_params(colors=text)
        ax.xaxis.label.set_color(text)
        ax.yaxis.label.set_color(text)
        if hasattr(ax, "zaxis") and ax.zaxis is not None:
            ax.zaxis.label.set_color(text)
        if hasattr(ax, "spines"):
            for spine in ax.spines.values():
                spine.set_color(self.palette["border"])
        if ax.name == "polar":
            self._apply_polar_orientation(ax)

    def _style_plot_axes(self) -> None:
        self.plot_figure.set_facecolor(self._current_plot_bg())
        self._style_axes(self.plot_ax)

    def _plot_grid_enabled(self) -> bool:
        checkbox = getattr(self, "chk_plot_grid_visible", None)
        if checkbox is None:
            return True
        return bool(checkbox.isChecked())

    def _current_plot_bg(self) -> str:
        return self.plot_bg_color or self.palette["panel_bg"]

    def _current_plot_grid(self) -> str:
        return self.plot_grid_color or self.palette["grid"]

    def _current_plot_text(self) -> str:
        return self.plot_text_color or self.palette["text"]

    def _apply_plot_theme(self) -> None:
        self.plot_figure.set_facecolor(self._current_plot_bg())
        axes = self.plot_axes or [self.plot_ax]
        for ax in axes:
            self._style_axes(ax)
            legend = ax.get_legend()
            if legend is not None:
                self._configure_legend(legend, ax)
                for text in legend.get_texts():
                    text.set_color(self._current_plot_text())
                legend.get_frame().set_facecolor(self._current_plot_bg())
                legend.get_frame().set_edgecolor(self._current_plot_grid())
        for colorbar in self.plot_colorbars:
            label_text = colorbar.ax.get_ylabel() or self._rcs_axis_label()
            colorbar.set_label(label_text, color=self._current_plot_text())
            colorbar.ax.tick_params(colors=self._current_plot_text())
            for label in colorbar.ax.get_yticklabels():
                label.set_color(self._current_plot_text())
        self.plot_canvas.draw_idle()

    def _apply_colorbar_ticks(self, colorbar) -> None:
        zstep = self.spin_plot_zstep.value()
        if zstep <= 0.0:
            return
        try:
            vmin, vmax = colorbar.mappable.get_clim()
        except Exception:
            return
        if vmin is None or vmax is None:
            return
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        ticks = np.arange(vmin, vmax + zstep * 0.5, zstep)
        if ticks.size == 0:
            return
        colorbar.set_ticks(ticks)

    def _choose_plot_color(self, which: str) -> None:
        if which == "bg":
            current = self._current_plot_bg()
            title = "Select Plot Background Color"
        elif which == "grid":
            current = self._current_plot_grid()
            title = "Select Plot Grid Color"
        else:
            current = self._current_plot_text()
            title = "Select Plot Text Color"
        color = QColorDialog.getColor(QColor(current), self, title)
        if not color.isValid():
            return
        if which == "bg":
            self.plot_bg_color = color.name()
        elif which == "grid":
            self.plot_grid_color = color.name()
        else:
            self.plot_text_color = color.name()
        self._update_plot_color_buttons()
        self._apply_plot_theme()

    def _update_plot_color_buttons(self) -> None:
        self.btn_plot_bg.setStyleSheet(f"background: {self._current_plot_bg()};")
        self.btn_plot_grid.setStyleSheet(f"background: {self._current_plot_grid()};")
        self.btn_plot_text.setStyleSheet(f"background: {self._current_plot_text()};")

    def _remove_colorbar(self) -> None:
        if not self.plot_colorbars:
            return
        for colorbar in self.plot_colorbars:
            try:
                if colorbar.ax is not None:
                    colorbar.remove()
            except Exception:
                pass
        self.plot_colorbars = []

    def _ensure_axes(self, projection: str) -> None:
        desired = "polar" if projection == "polar" else "rectilinear"
        if self.plot_ax.name == desired and self.plot_axes is None:
            return
        self._remove_colorbar()
        self.plot_figure.clear()
        if desired == "polar":
            self.plot_ax = self.plot_figure.add_subplot(111, projection="polar")
        else:
            self.plot_ax = self.plot_figure.add_subplot(111)
        self.plot_axes = None
        self._style_plot_axes()

    def _clear_plot(self) -> None:
        self.plot_ax.clear()
        self._remove_colorbar()
        self.plot_axes = None
        self._style_plot_axes()
        self._apply_plot_limits()

    def _single_selection_index(self, widget: QListWidget, label: str) -> int | None:
        selected = sorted(self._selected_indices(widget))
        if len(selected) != 1:
            count = len(selected)
            if count == 0:
                msg = f"Select 1 {label} to plot."
            else:
                msg = f"Select exactly 1 {label} (selected {count})."
            self.status.showMessage(msg)
            return None
        return selected[0]

    def _single_selection_value(self, widget: QListWidget, label: str):
        values = self._selected_values(widget)
        if len(values) != 1:
            count = len(values)
            if count == 0:
                msg = f"Select 1 {label} to plot."
            else:
                msg = f"Select exactly 1 {label} (selected {count})."
            self.status.showMessage(msg)
            return None
        return values[0]

    @staticmethod
    def _button_checked(button: QToolButton | None) -> bool:
        return bool(button.isChecked()) if button is not None else False

    def _apply_plot_limits(self) -> None:
        xmin = self.spin_plot_xmin.value()
        xmax = self.spin_plot_xmax.value()
        ymin = self.spin_plot_ymin.value()
        ymax = self.spin_plot_ymax.value()
        xstep = self.spin_plot_xstep.value()
        ystep = self.spin_plot_ystep.value()
        axes = self.plot_axes or [self.plot_ax]
        for ax in axes:
            ax.set_autoscale_on(False)
            if ax.name == "polar":
                ax.set_thetamin(xmin)
                ax.set_thetamax(xmax)
            else:
                ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            if xstep > 0.0:
                if ax.name == "polar":
                    ax.set_thetagrids(np.arange(xmin, xmax + xstep * 0.5, xstep))
                else:
                    ax.set_xticks(np.arange(xmin, xmax + xstep * 0.5, xstep))
            if ystep > 0.0:
                ax.set_yticks(np.arange(ymin, ymax + ystep * 0.5, ystep))
            if xstep <= 0.0 and ax.name == "polar":
                ax.set_thetagrids(np.arange(xmin, xmax + 45.0, 45.0))
        self.plot_canvas.draw_idle()

    def _fit_both(self) -> None:
        if self.plot_ax.name == "polar":
            self._fit_y()
            return
        self._fit_x()
        self._fit_y()

    def _copy_plot(self) -> None:
        from PySide6.QtWidgets import QApplication
        pixmap = self.plot_canvas.grab()
        QApplication.clipboard().setPixmap(pixmap)
        self.status.showMessage("Plot copied to clipboard.")

    def _effective_colormap(self) -> str:
        name = self.combo_colormap.currentText()
        if self.chk_colormap_invert.isChecked():
            name = name + "_r"
        return name

    def _isar_window(self, n: int) -> np.ndarray:
        if n <= 1:
            return np.ones(n)
        name = self.combo_isar_window.currentText()
        if name == "Hamming":
            return np.hamming(n)
        if name == "Blackman":
            return np.blackman(n)
        if name == "Rectangular":
            return np.ones(n)
        return np.hanning(n)

    def _on_phase_toggled(self) -> None:
        self._on_polarization_selection_changed()
        self._maybe_autoplot()

    def _on_isar_window_changed(self) -> None:
        if self.last_plot_mode == "isar_image":
            self._plot_isar_image()
        elif self.last_plot_mode == "isar_3d":
            self._plot_isar_3d()

    def _fit_polar_x_range(self) -> tuple[float, float]:
        theta_values: list[np.ndarray] = []
        for line in self.plot_ax.lines:
            try:
                x = np.asarray(line.get_xdata(), dtype=float)
            except Exception:
                continue
            if x.size == 0:
                continue
            finite = x[np.isfinite(x)]
            if finite.size:
                theta_values.append(np.degrees(finite))

        if not theta_values:
            xmin, xmax = np.degrees(self.plot_ax.get_xlim())
            xmin = float(xmin)
            xmax = float(xmax)
            if not np.isfinite(xmin) or not np.isfinite(xmax) or np.isclose(xmin, xmax):
                return -180.0, 180.0
            if xmax < xmin:
                xmax += 360.0
            if (xmax - xmin) >= 359.0:
                return -180.0, 180.0
            return xmin, xmax

        theta = np.mod(np.concatenate(theta_values), 360.0)
        theta.sort()
        if theta.size == 1:
            center = float(theta[0])
            return center - 5.0, center + 5.0

        wrapped = np.concatenate([theta, [theta[0] + 360.0]])
        gaps = np.diff(wrapped)
        gap_idx = int(np.argmax(gaps))
        largest_gap = float(gaps[gap_idx])
        span = 360.0 - largest_gap
        if span >= 359.0:
            return -180.0, 180.0

        start = float(theta[(gap_idx + 1) % theta.size])
        end = start + span
        pad = max(1.0, 0.03 * span)
        xmin = start - pad
        xmax = end + pad
        if (xmax - xmin) >= 359.0:
            return -180.0, 180.0

        while xmin > 180.0:
            xmin -= 360.0
            xmax -= 360.0
        while xmin <= -180.0:
            xmin += 360.0
            xmax += 360.0
        return xmin, xmax

    def _fit_polar_y_range(self) -> tuple[float, float]:
        radial_values: list[np.ndarray] = []
        for line in self.plot_ax.lines:
            try:
                y = np.asarray(line.get_ydata(), dtype=float)
            except Exception:
                continue
            if y.size == 0:
                continue
            finite = y[np.isfinite(y)]
            if finite.size:
                radial_values.append(finite)

        if radial_values:
            radial = np.concatenate(radial_values)
            ymin = float(np.nanmin(radial))
            ymax = float(np.nanmax(radial))
        else:
            ymin, ymax = self.plot_ax.get_ylim()
            ymin = float(ymin)
            ymax = float(ymax)

        if not np.isfinite(ymin) or not np.isfinite(ymax):
            return -1.0, 1.0
        if np.isclose(ymin, ymax):
            pad = max(1.0, abs(ymin) * 0.05)
            ymin -= pad
            ymax += pad
        return ymin, ymax

    def _fit_x(self) -> None:
        if self.plot_ax.name == "polar":
            return

        self.plot_ax.set_autoscale_on(True)
        self.plot_ax.relim()
        self.plot_ax.autoscale_view(scalex=True, scaley=False)
        xmin, xmax = self.plot_ax.get_xlim()
        self.spin_plot_xmin.blockSignals(True)
        self.spin_plot_xmax.blockSignals(True)
        self.spin_plot_xmin.setValue(float(xmin))
        self.spin_plot_xmax.setValue(float(xmax))
        if self.spin_plot_xstep.value() > 0.0:
            self.spin_plot_xstep.blockSignals(True)
            self.spin_plot_xstep.setValue(0.0)
            self.spin_plot_xstep.blockSignals(False)
        self.spin_plot_xmin.blockSignals(False)
        self.spin_plot_xmax.blockSignals(False)
        self._apply_plot_limits()

    def _fit_y(self) -> None:
        if self.plot_ax.name == "polar":
            ymin, ymax = self._fit_polar_y_range()
            self.spin_plot_ymin.blockSignals(True)
            self.spin_plot_ymax.blockSignals(True)
            self.spin_plot_ymin.setValue(float(ymin))
            self.spin_plot_ymax.setValue(float(ymax))
            if self.spin_plot_ystep.value() > 0.0:
                self.spin_plot_ystep.blockSignals(True)
                self.spin_plot_ystep.setValue(0.0)
                self.spin_plot_ystep.blockSignals(False)
            self.spin_plot_ymin.blockSignals(False)
            self.spin_plot_ymax.blockSignals(False)
            axes = self.plot_axes or [self.plot_ax]
            for ax in axes:
                ax.set_autoscale_on(False)
                ax.set_ylim(ymin, ymax)
            self.plot_canvas.draw_idle()
            return
        else:
            self.plot_ax.set_autoscale_on(True)
            self.plot_ax.relim()
            self.plot_ax.autoscale_view(scalex=False, scaley=True)
            ymin, ymax = self.plot_ax.get_ylim()
        self.spin_plot_ymin.blockSignals(True)
        self.spin_plot_ymax.blockSignals(True)
        self.spin_plot_ymin.setValue(float(ymin))
        self.spin_plot_ymax.setValue(float(ymax))
        if self.spin_plot_ystep.value() > 0.0:
            self.spin_plot_ystep.blockSignals(True)
            self.spin_plot_ystep.setValue(0.0)
            self.spin_plot_ystep.blockSignals(False)
        self.spin_plot_ymin.blockSignals(False)
        self.spin_plot_ymax.blockSignals(False)
        self._apply_plot_limits()

    def _collect_azimuth_series(
        self,
        dataset: RcsGrid,
        dataset_name: str,
        az_values_sel: list,
        elev_values_sel: list,
        freq_values_sel: list,
        pol_value_sel,
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, str]]] | None:
        az_indices = self._indices_for_values(dataset.azimuths, az_values_sel)
        elev_indices = self._indices_for_values(dataset.elevations, elev_values_sel)
        freq_indices = self._indices_for_values(dataset.frequencies, freq_values_sel)
        pol_indices = self._indices_for_values(dataset.polarizations, [pol_value_sel], tol=0.0)
        if az_indices is None or elev_indices is None or freq_indices is None or pol_indices is None:
            return None

        az_values = dataset.azimuths[az_indices]
        order = np.argsort(az_values)
        az_values = az_values[order]
        pol_value = dataset.polarizations[pol_indices[0]]
        series: list[tuple[np.ndarray, str]] = []
        for freq_idx in freq_indices:
            freq_value = dataset.frequencies[freq_idx]
            for elev_idx in elev_indices:
                elev_value = dataset.elevations[elev_idx]
                if self._button_checked(self.btn_phase):
                    rcs_values = dataset.rcs[az_indices, elev_idx, freq_idx, pol_indices[0]]
                else:
                    rcs_values = dataset.rcs_power[az_indices, elev_idx, freq_idx, pol_indices[0]]
                rcs_display = self._rcs_display_values(dataset, rcs_values)
                rcs_display = rcs_display[order]
                label = (
                    f"{dataset_name} | Pol {pol_value}, Freq {freq_value} GHz, El {elev_value} deg"
                )
                series.append((rcs_display, label))

        return az_values, series

    def _legend_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {"loc": "upper right"}
        if self.last_plot_mode == "compare":
            kwargs["fontsize"] = 8
        return kwargs

    def _configure_legend(self, legend, ax=None) -> None:
        if legend is None:
            return
        if ax is None:
            ax = self.plot_ax
        try:
            legend.set_loc("upper right")
            legend.set_bbox_to_anchor((0.98, 0.98), transform=ax.transAxes)
        except Exception:
            pass
        try:
            legend.set_draggable(True, use_blit=True, update="loc")
        except TypeError:
            try:
                legend.set_draggable(True, use_blit=True)
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _format_hover_number(value) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "--"
        if not np.isfinite(number):
            return "--"
        magnitude = abs(number)
        if magnitude >= 1e4 or (0.0 < magnitude < 1e-3):
            return f"{number:.4e}"
        return f"{number:.6f}"

    @staticmethod
    def _cursor_data_to_scalar(data) -> float | None:
        if data is None:
            return None
        try:
            values = np.asarray(data)
            if values.size == 0:
                return None
            if np.iscomplexobj(values):
                values = np.real(values)
            flat = np.asarray(values, dtype=float).ravel()
        except Exception:
            return None
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return None
        return float(finite[0])

    def _hover_z_from_axes(self, ax, event) -> float | None:
        artists = []
        artists.extend(reversed(getattr(ax, "collections", [])))
        artists.extend(reversed(getattr(ax, "images", [])))
        for artist in artists:
            getter = getattr(artist, "get_cursor_data", None)
            if getter is None:
                continue
            try:
                value = self._cursor_data_to_scalar(getter(event))
            except Exception:
                continue
            if value is not None:
                return value
        return None

    def _nearest_3d_hover_point(self, ax, event) -> tuple[float, float, float] | None:
        try:
            from mpl_toolkits.mplot3d import proj3d
        except Exception:
            return None
        if getattr(event, "x", None) is None or getattr(event, "y", None) is None:
            return None

        view_key = (
            round(float(getattr(ax, "elev", 0.0)), 3),
            round(float(getattr(ax, "azim", 0.0)), 3),
            tuple(np.round(np.asarray(ax.get_xlim3d(), dtype=float), 6)),
            tuple(np.round(np.asarray(ax.get_ylim3d(), dtype=float), 6)),
            tuple(np.round(np.asarray(ax.get_zlim3d(), dtype=float), 6)),
        )
        cache = getattr(ax, "_grim_hover_cache", None)
        if not isinstance(cache, dict) or cache.get("view_key") != view_key:
            xyz_chunks: list[np.ndarray] = []
            xy_chunks: list[np.ndarray] = []
            for artist in getattr(ax, "collections", []):
                offsets3d = getattr(artist, "_offsets3d", None)
                if offsets3d is None:
                    continue
                try:
                    xs = np.asarray(offsets3d[0], dtype=float).ravel()
                    ys = np.asarray(offsets3d[1], dtype=float).ravel()
                    zs = np.asarray(offsets3d[2], dtype=float).ravel()
                except Exception:
                    continue
                finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
                if not np.any(finite):
                    continue
                xs = xs[finite]
                ys = ys[finite]
                zs = zs[finite]
                x2d, y2d, _ = proj3d.proj_transform(xs, ys, zs, ax.get_proj())
                finite_2d = np.isfinite(x2d) & np.isfinite(y2d)
                if not np.any(finite_2d):
                    continue
                xs = xs[finite_2d]
                ys = ys[finite_2d]
                zs = zs[finite_2d]
                x2d = x2d[finite_2d]
                y2d = y2d[finite_2d]
                xy_pixels = ax.transData.transform(np.column_stack([x2d, y2d]))
                xyz_chunks.append(np.column_stack([xs, ys, zs]))
                xy_chunks.append(xy_pixels)
            if not xyz_chunks or not xy_chunks:
                return None
            cache = {
                "view_key": view_key,
                "xyz": np.vstack(xyz_chunks),
                "xy": np.vstack(xy_chunks),
            }
            setattr(ax, "_grim_hover_cache", cache)

        xy_pixels = cache.get("xy")
        xyz_points = cache.get("xyz")
        if xy_pixels is None or xyz_points is None or len(xy_pixels) == 0:
            return None

        distances = np.hypot(xy_pixels[:, 0] - event.x, xy_pixels[:, 1] - event.y)
        finite = np.isfinite(distances)
        if not np.any(finite):
            return None
        idx = int(np.argmin(np.where(finite, distances, np.inf)))
        if distances[idx] > 24.0:
            return None
        x_val, y_val, z_val = xyz_points[idx]
        return float(x_val), float(y_val), float(z_val)

    def _reset_hover_readout(self, hover_readout=None) -> None:
        label = hover_readout or getattr(self, "hover_readout", None)
        if label is None:
            return
        label.setText("x: --   y: --")

    def _on_plot_hover(self, event, hover_readout=None) -> None:
        label = hover_readout or getattr(self, "hover_readout", None)
        if label is None:
            return
        ax = getattr(event, "inaxes", None)
        if ax is None:
            self._reset_hover_readout(label)
            return
        if ax.name == "3d":
            point = self._nearest_3d_hover_point(ax, event)
            if point is None:
                label.setText("x: --   y: --\nz: --")
                return
            x_val, y_val, z_val = point
            label.setText(
                f"x: {self._format_hover_number(x_val)}   y: {self._format_hover_number(y_val)}\n"
                f"z: {self._format_hover_number(z_val)}"
            )
            return

        x_val = getattr(event, "xdata", None)
        y_val = getattr(event, "ydata", None)
        if (
            x_val is None
            or y_val is None
            or not np.isfinite(x_val)
            or not np.isfinite(y_val)
        ):
            self._reset_hover_readout(label)
            return

        z_val = self._hover_z_from_axes(ax, event)
        if z_val is None:
            label.setText(
                f"x: {self._format_hover_number(x_val)}   y: {self._format_hover_number(y_val)}"
            )
            return
        label.setText(
            f"x: {self._format_hover_number(x_val)}   y: {self._format_hover_number(y_val)}\n"
            f"z: {self._format_hover_number(z_val)}"
        )

    def _update_legend_visibility(self) -> None:
        legend = self.plot_ax.get_legend()
        if self.chk_plot_legend.isChecked():
            if legend is None:
                handles, labels = self.plot_ax.get_legend_handles_labels()
                if handles:
                    legend = self.plot_ax.legend(**self._legend_kwargs())
                else:
                    return
            legend.set_visible(True)
            self._configure_legend(legend, self.plot_ax)
        else:
            if legend is not None:
                legend.set_visible(False)
        self.plot_canvas.draw_idle()

    def _plot_azimuth_rect(self) -> None:
        azimuth_rect_mode.render(self)

    def _plot_azimuth_polar(self) -> None:
        azimuth_polar_mode.render(self)

    def _plot_frequency(self) -> None:
        frequency_mode.render(self)

    def _plot_elevation_sweep(self) -> None:
        elevation_sweep_mode.render(self)

    def _plot_isar_image(self) -> None:
        isar_mode.render(self)

    def _plot_isar_3d(self) -> None:
        isar_3d_mode.render(self)

    def _plot_waterfall(self) -> None:
        waterfall_mode.render(self)

    def _plot_compare(self) -> None:
        compare_mode.render(self)

    def _ensure_compare_axes(self):
        """Return (top_ax, res_ax) for the 2-panel compare layout, recreating if needed."""
        if (
            self.plot_axes is not None
            and len(self.plot_axes) == 1
            and len(self.plot_figure.axes) == 2
        ):
            return self.plot_figure.axes[0], self.plot_figure.axes[1]
        self._remove_colorbar()
        self.plot_figure.clear()
        top_ax, res_ax = self.plot_figure.subplots(
            2, 1, sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
        )
        self.plot_ax = top_ax
        # plot_axes = [top_ax] so _apply_plot_limits only touches the top axis
        # (x-limits propagate automatically via sharex; residual y auto-scales)
        self.plot_axes = [top_ax]
        self.plot_figure.set_facecolor(self._current_plot_bg())
        return top_ax, res_ax
