#pyside6-uic Visu.ui -o Visu.py
from pathlib import Path, PurePosixPath
import subprocess
import sys

from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QSlider, QPushButton, QCheckBox, QSizePolicy, QButtonGroup
)
from Visu import Ui_MainWindow
from State_Holder import StateHolder
from spline_editor import SplineEditor
from zoom_view import ZoomPanGraphicsView
import torch
import json
import yaml
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.collections import PolyCollection, LineCollection

from contextlib import contextmanager

from helper_functions import *
from preparation import *
from neural_network import *


@contextmanager
def block_signals(*widgets):
    try:
        for w in widgets: w.blockSignals(True)
        yield
    finally:
        for w in widgets: w.blockSignals(False)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # because we inherit Ui_MainWindow
        self._settings = QtCore.QSettings()

        h_layout = QHBoxLayout(self.figureWidget)
        # Matplotlib Figure
        self.figure = Figure(figsize=(100, 100))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal')
        #self.ax.grid()
        self.canvas.draw()
        h_layout.addWidget(self.canvas)

        self.canvas.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        self.canvas.updateGeometry()
        h_layout.addWidget(self.canvas, stretch=1)

        # set up a StateHolder
        self.state = StateHolder()

        self.graphicsView = ZoomPanGraphicsView.replace(self.graphicsView)

        self.editor = SplineEditor(self.graphicsView, self.state, point_radius_px=3.0)
        self.editor.controlPointsChanged.connect(self.redraw)

        self.pc: PolyCollection = None
        self.cax = inset_axes(
                        self.ax,
                        width="30%",    # same as before
                        height="100%",
                        loc="upper right",
                        bbox_to_anchor=(0.75, 0.1, 0.15, 0.8),
                        bbox_transform=self.ax.transAxes
                        )
        self.cax.set_visible(False)

        self.cbar = self.figure.colorbar(self.pc,
                                         cax=self.cax,
                                         orientation="vertical")

        self.stress_checkbox.stateChanged.connect(self.on_stress_checkbox_changed)
        self.vmin_spin.valueChanged.connect(self.update_colorbar_limits)
        self.vmax_spin.valueChanged.connect(self.update_colorbar_limits)

        # Mouse interaction
        self.pressing = False
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        # --- always-on left-drag pan using Matplotlib's built-in mechanism ---
        self._pan_ax = None
        self._pan_btn = 1

        self.canvas.mpl_connect("button_press_event",
                                lambda e: (setattr(self, "_pan_ax", e.inaxes), e.inaxes.start_pan(e.x, e.y, e.button))
                                if (e.button == self._pan_btn and e.inaxes is not None) else None)

        self.canvas.mpl_connect("motion_notify_event",
                                lambda e: (self._pan_ax.drag_pan(self._pan_btn, e.key, e.x, e.y),
                                           self.canvas.draw_idle())
                                if (self._pan_ax is not None) else None)

        self.canvas.mpl_connect("button_release_event",
                                lambda e: (self._pan_ax.end_pan(), setattr(self, "_pan_ax", None))
                                if (e.button == self._pan_btn and self._pan_ax is not None) else None)

        self.xy_pad.pointChanged.connect(self.on_pad_changed)
        self.spin_x.valueChanged.connect(self.on_spin_x_changed)
        self.spin_y.valueChanged.connect(self.on_spin_y_changed)
        self.spin_poisson.valueChanged.connect(self.trigger_redraw)

        self.check_solution_button.clicked.connect(self.check_solution)

        # Load defaults
        with open('boundary_conditions.yaml', "r", encoding="utf-8") as f:
            self.state.boundary_conditions = yaml.safe_load(f) or {}
            self.set_bc(self.state.boundary_conditions)

        self.state.G = load_Bspline('input_geometries/simple_5.xml')
        self.editor.set_base_spline(self.state.G)

        model = torch.load("models/network_[1, 1]_r_8_1000.pth", weights_only=False)
        self.state.pinn = Pinn(model)

        self.trigger_redraw()

    def on_stress_checkbox_changed(self):
        self.toggle_colorbar()
        self.redraw()


    def on_pad_changed(self, x, y):
        with block_signals(self.spin_x, self.spin_y):
            self.spin_x.setValue(x)
            self.spin_y.setValue(y)
        self.trigger_redraw()

    def on_spin_x_changed(self, x):
        with block_signals(self.xy_pad):
            self.xy_pad.setPoint(x, self.spin_y.value())
        self.trigger_redraw()

    def on_spin_y_changed(self, y):
        with block_signals(self.xy_pad):
            self.xy_pad.setPoint(self.spin_x.value(), y)
        self.trigger_redraw()

    def trigger_redraw(self):
        self.prepare_boundary_conditions()
        self.redraw()

    def update_colorbar_limits(self):
        if not self.pc:
            return
        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()
        # guard against vmin >= vmax
        if vmin >= vmax:
            return
        self.pc.set_clim(vmin, vmax)
        if self.cbar:
            # update the bar to reflect new limits
            self.cbar.update_normal(self.pc)
        self.canvas.draw()


    def set_bc(self, bc):
        self.state.dirichlet_boundary = bc["dirichlet"]
        self.state.neumann_boundary = bc["neumann"]       # left empty for now

    def is_active(self, side):
        if side == 'a':
            return self.bnd_a.isChecked()
        elif side == 'b':
            return self.bnd_b.isChecked()
        elif side == 'c':
            return self.bnd_c.isChecked()
        elif side == 'd':
            return self.bnd_d.isChecked()


    def prepare_boundary_conditions(self):
        x = self.spin_x.value()
        y = self.spin_y.value()
        for side in self.state.dirichlet_boundary.keys():
            if self.is_active(side):
                for key in self.state.dirichlet_boundary[side].keys():
                    self.state.dirichlet_boundary[side][key][0] = x
                    self.state.dirichlet_boundary[side][key][1] = y



    def toggle_colorbar(self):
        self.cax.set_visible(self.stress_checkbox.isChecked())
        self.redraw()

    def on_press(self, event):
        if event.inaxes == self.ax:
            self.pressing = True
            #self.update_from_event(event)

    def on_release(self, event):
        self.pressing = False

    def on_motion(self, event):
        if self.pressing and event.inaxes == self.ax:
            pass#self.update_from_event(event)

    def on_scroll(self, event):
        """Zoom in or out around the mouse pointer."""
        base_scale = 1.2
        # get current limits
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return

        if event.button == 'up':
            # zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # zoom out
            scale_factor = base_scale
        else:
            return

        # compute new limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([
            xdata - new_width * relx,
            xdata + new_width * (1 - relx)
        ])
        self.ax.set_ylim([
            ydata - new_height * rely,
            ydata + new_height * (1 - rely)
        ])

        self.canvas.draw_idle()


    def update_from_event(self, event):
        # Clamp to [0,2]
        x = min(max(event.xdata, -2), 2)
        y = min(max(event.ydata, -0.5), 3)
        # Round to nearest 0.05
        step = 0.05
        x = round(x/step) * step
        y = round(y/step) * step

        # Block signals to avoid recursive redraw
        self.spin_x.blockSignals(True)
        self.spin_y.blockSignals(True)
        self.spin_x.setValue(x)
        self.spin_y.setValue(y)
        self.spin_x.blockSignals(False)
        self.spin_y.blockSignals(False)

        self.redraw()


    def create_color_data(self, S, u):
        po = self.spin_poisson.value()
        E = 1
        mu_ = E / (2 * (1 + po))
        lambda_ = po * E / ((1 + po) * (1 - 2 * po))

        mesh = export_to_paraview(S, u, lambda_, mu_)
        # Get all points of the deformed mesh
        # mesh_points = mesh.points[:, :2]
        deformed_mesh = S.extract.faces([32, 32])
        # Vertice coordinates and connectivity
        mesh_points = deformed_mesh.vertices
        # Get mesh faces (consisting of four points each)
        faces = mesh.cells_dict["quad"]
        # Get the point values of a metric
        if self.stress_checkbox.isChecked():
            point_values = mesh.point_data["von_Mises"]
        # Get the value of a face by averaging over corner point values
        face_values = point_values[faces].mean(axis=1)
        # Get coordinates of the faces
        face_coordinates = [mesh_points[i] for i in faces]
        # Generate plot data
        return PolyCollection(face_coordinates, array=face_values, cmap="viridis", linewidths=0)


    def redraw(self):
        for line in list(self.ax.get_lines()):
            line.remove()
        for col in list(self.ax.collections):
            if isinstance(col, PolyCollection):
                col.remove()
            elif isinstance(col, (LineCollection, PolyCollection)):
                col.remove()

        u = self.state.pinn.evaluate(self.state.base, self.state.dirichlet_boundary, self.state.neumann_boundary, self.spin_poisson.value())
        S = splinepy.BSpline(degrees=self.state.base.degrees, knot_vectors=self.state.base.knot_vectors,
                             control_points=self.state.base.control_points + u.control_points)
        plot_Bspline(S, self.ax)

        if self.stress_checkbox.isChecked():
            self.pc = self.create_color_data(S, u)
            self.pc.set_clim(self.vmin_spin.value(), self.vmax_spin.value())
            self.ax.add_collection(self.pc)
        self.canvas.draw()



    def check_solution(self):
        def run_experiment(x, y, p):
            """
            base structure of this function written by ChatGPT
            """
            if sys.platform.startswith("linux"):
                proj = "/home/chg/Programming/IGN_neoHook/70_Visualization"
                exe = f"{proj}/Experiment_4_nonlinear"
                xml = f"{proj}/gismo_check.xml"
                out = proj if proj.endswith("/") else proj + "/"
                subprocess.run(
                    [
                        exe, "-f", xml,
                        "-x", str(x), "-y", str(y), "-p", str(p),
                        "-A", "2", "-B", "2", "-E", "1",
                        "-P", out, "-R", "0.5",
                    ],
                    check=True,
                    cwd=proj,  # Paraview output goes here
                )
            else:
                # Windows → run Linux binary inside WSL, writing onto the Windows folder
                proj = PurePosixPath(
                    "/mnt/c/Users/ChristianGollmann/Programming/IGN_neoHook/70_Visualization")
                exe = proj / "Experiment_4_nonlinear"
                xml = proj / "gismo_check.xml"
                out = (str(proj) + "/")  # IMPORTANT: trailing slash for -P
                print(f"checking solution with x: {x}, y: {y}, p: {p}")
                # ensure working dir is the project on /mnt/c so Paraview files land there
                wsl_cmd = (
                    f'"{exe}" -f "{xml}" '
                    f'-x {x} -y {y} -p {p} '
                    f'-A 2 -B 2 -E 1 '
                    f'-P "{out}" -R 0.5'
                )

                subprocess.run(
                    ["wsl", "--cd", str(proj), "bash", "-lc", wsl_cmd],
                    check=True
                )

        # export the base geometry to xml
        G = splinepy.BSpline(degrees=self.state.base.degrees, knot_vectors=self.state.base.knot_vectors,
                     control_points=self.state.base.control_points)
        splinepy.io.gismo.export("gismo_check.xml", G)
        # first calculate the gismo solution
        run_experiment(self.spin_x.value(), self.spin_y.value(), self.spin_poisson.value())

        # now load the gismo solution and plot it
        G = load_Bspline("geometry_gismo.xml")
        u = load_Bspline("displacement_gismo.xml")
        S = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors,
                             control_points=G.control_points + u.control_points)

        # Export the network solution as well for comparison reasons
        u_net = self.state.pinn.evaluate(self.state.base, self.state.dirichlet_boundary, self.state.neumann_boundary,
                                     self.spin_poisson.value())
        splinepy.io.gismo.export("net_solution.xml", u_net)
        plot_Bspline(S, self.ax, color='green')
        self.canvas.draw()





    # You can name handlers however you like…
    @Slot(bool)
    def on_actionGeometry_triggered(self, checked: bool = False):
        start_dir = self._settings.value("paths/last_geometry_dir", str(Path.home()))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Geometry",
            start_dir)
        if not path:
            return
        self._settings.setValue("paths/last_geometry_dir", str(Path(path).parent))
        try:
            self.statusBar().showMessage("Loading geometry …")
            self.state.G = load_Bspline(path)
            self.statusBar().showMessage("Geometry loaded")
            self.editor.set_base_spline(self.state.G)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Failed to load geometry", str(e))
            self.statusBar().clearMessage()
            return


    @Slot(bool)
    def on_actionModel_triggered(self, checked: bool = False):
        start_dir = self._settings.value("paths/last_model_dir", str(Path.home()))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open PyTorch model",
            start_dir)
        if not path:
            return
        self._settings.setValue("paths/last_model_dir", str(Path(path).parent))
        try:
            self.statusBar().showMessage("Loading model …")
            model = torch.load(path, weights_only=False)
            self.state.pinn = Pinn(model)
            self.statusBar().showMessage("Model loaded")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Failed to load model", str(e))
            self.statusBar().clearMessage()
            return


    @Slot(bool)
    def on_actionBoundary_Conditions_triggered(self, checked: bool = False):
        start_dir = self._settings.value("paths/last_bc_dir", str(Path.home()))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Boundary Conditions",
            start_dir)
        if not path:
            return
        self._settings.setValue("paths/last_bc_dir", str(Path(path).parent))
        try:
            self.statusBar().showMessage("Loading boundary conditions …")
            with open(path, "r", encoding="utf-8") as f:
                self.state.boundary_conditions = yaml.safe_load(f) or {}
                self.set_bc(self.state.boundary_conditions)
            self.statusBar().showMessage("Boundary conditions loaded")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Failed to load boundary conditions", str(e))
            self.statusBar().clearMessage()
            return

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()