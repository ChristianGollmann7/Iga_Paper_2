# spline_editor.py
from __future__ import annotations
from typing import Optional, Sequence, List
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import splinepy

class SplineEditor(QtCore.QObject):
    controlPointsChanged = QtCore.Signal() # final "committed" change

    def __init__(
        self,
        view: QtWidgets.QGraphicsView,
        state,
        *,
        point_radius_px: float = 7.0,
        step_ratio: float = 0.005,   # 0.5% of initial span
        min_step: float = 1e-6,
        coalesce_ms: int = 30,       # live redraw rate limit (~33 fps)
    ) -> None:
        super().__init__(view)
        self.view = view
        self.state = state
        self.scene = QtWidgets.QGraphicsScene(self.view)
        self.view.setScene(self.scene)

        self._handles: List[_ControlPointItem] = []
        self._suppress = False

        # Snapping
        self._step_ratio = float(step_ratio)
        self._min_step = float(min_step)
        self._grid_step = 0.0
        self._grid_anchor_x = 0.0
        self._grid_anchor_y = 0.0
        self._eps = 1e-12

        # Batching/coalescing
        self._batching = False          # true when multi-select drag active
        self._active_drags = 0          # how many handles are currently pressed
        self._dirty_during_drag = False # did anything move since last redraw?
        self._coalesce_ms = int(coalesce_ms)
        self._redraw_timer = QtCore.QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self._do_coalesced_redraw)

        # Appearance
        self.point_radius_px = float(point_radius_px)
        self.fill_normal   = QtGui.QBrush(QtGui.QColor(0, 190, 255))
        self.fill_selected = QtGui.QBrush(QtGui.QColor(255, 215, 0))

        # Optional background iso-lines
        self._iso_items: List[QtWidgets.QGraphicsPathItem] = []
        self._iso_pen = QtGui.QPen(QtGui.QColor(120, 120, 120), 1)
        self._iso_pen.setCosmetic(True)

        # View niceties
        self.view.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        self.view.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
        self.view.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.scene.setItemIndexMethod(QtWidgets.QGraphicsScene.ItemIndexMethod.NoIndex)

    # ----------------- Public API -----------------

    def set_base_spline(self, G) -> None:
        self.state.base = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors, control_points=G.control_points)
        self.state.modified = splinepy.BSpline(degrees=G.degrees, knot_vectors=G.knot_vectors, control_points=G.control_points)
        self.set_control_points(G.control_points)
        self.update_surface_iso()



    def set_control_points(self, control_points: np.ndarray) -> None:
        cps = np.asarray(control_points, dtype=float)
        if cps.ndim != 2 or cps.shape[1] < 2:
            raise ValueError("control_points must be (N,2+) array")
        self.state.control_points = cps[:, :2].copy()

        # Compute snap grid
        x = self.state.control_points[:, 0]; y = self.state.control_points[:, 1]
        x0, x1 = float(x.min()), float(x.max())
        y0, y1 = float(y.min()), float(y.max())
        span = max(x1 - x0, y1 - y0)
        self._grid_step = max(span * self._step_ratio, self._min_step)
        self._grid_anchor_x = x0
        self._grid_anchor_y = y0

        # remove old handles
        for h in self._handles:
            self.scene.removeItem(h)
        self._handles.clear()

        # build handles
        for i in range(len(self.state.control_points)):
            px = self.state.control_points[i][0]
            py = self.state.control_points[i][1]
            h = _ControlPointItem(
                index=i,
                center=QtCore.QPointF(float(px), float(py)),
                radius_px=self.point_radius_px,
                normal_brush=self.fill_normal,
                selected_brush=self.fill_selected,
            )
            h.setZValue(10)
            h.positionChanged.connect(self._on_handle_moved)
            h.dragStarted.connect(self._on_handle_drag_started)
            h.dragFinished.connect(self._on_handle_drag_finished)
            self.scene.addItem(h)
            self._handles.append(h)

        self.fit_to_control_points()

    def get_control_points(self) -> np.ndarray:
        if self.state.control_points is None:
            return np.empty((0, 2), dtype=float)
        return self.state.control_points

    def fit_to_control_points(self, padding_ratio: float = 0.01) -> None:
        cps = self.state.control_points
        if cps is None or cps.size == 0:
            return
        x = cps[:, 0]; y = cps[:, 1]
        x0, x1 = float(x.min()), float(x.max())
        y0, y1 = float(y.min()), float(y.max())
        w = x1 - x0; h = y1 - y0
        if w <= 0: w = 1.0
        if h <= 0: h = 1.0
        pad = max(w, h) * padding_ratio
        rect = QtCore.QRectF(x0 - pad, y0 - pad, w + 2*pad, h + 2*pad)
        self.view.resetTransform()
        self.scene.setSceneRect(rect)
        self.view.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    # ----------------- Drag batching -----------------

    def _on_handle_drag_started(self, _idx: int) -> None:
        self._active_drags += 1
        selected_count = sum(1 for h in self._handles if h.isSelected())
        if selected_count > 1:
            self._batching = True
            self._dirty_during_drag = False

    def _on_handle_drag_finished(self, _idx: int) -> None:
        self._active_drags = max(0, self._active_drags - 1)
        if self._active_drags == 0:
            # flush any pending coalesced redraw
            if self._redraw_timer.isActive():
                self._redraw_timer.stop()
                self._do_coalesced_redraw()
            # if we were batching and things moved, emit final signal once
            if self._batching and self._dirty_during_drag:
                self.controlPointsChanged.emit()
            self._batching = False
            self._dirty_during_drag = False

    # ----------------- Move handling -----------------

    def _on_handle_moved(self, index: int, pos: QtCore.QPointF) -> None:
        if self._suppress or self.state.control_points is None:
            return

        sx = self._snap_value(pos.x(), self._grid_anchor_x, self._grid_step)
        sy = self._snap_value(pos.y(), self._grid_anchor_y, self._grid_step)
        snapped = QtCore.QPointF(sx, sy)

        # move handle to snapped position if needed
        if (abs(snapped.x() - pos.x()) > self._eps) or (abs(snapped.y() - pos.y()) > self._eps):
            self._suppress = True
            self._handles[index].set_center(snapped)
            self._suppress = False

        oldx, oldy = self.state.control_points[index]
        if abs(oldx - sx) <= self._eps and abs(oldy - sy) <= self._eps:
            return

        # update model
        self.state.base.control_points[index, 0] = sx
        self.state.base.control_points[index, 1] = sy


        # schedule a coalesced redraw (live preview without spamming)
        self._dirty_during_drag = True
        self._schedule_redraw()

        # For single-point drags, still emit immediately (snappy UX)
        if not self._batching:
            self.controlPointsChanged.emit()

    # ----------------- Coalesced redraw -----------------

    def _schedule_redraw(self) -> None:
        # restart the timer to coalesce bursts of moves
        self._redraw_timer.start(self._coalesce_ms)

    def _do_coalesced_redraw(self) -> None:
        if not self._dirty_during_drag:
            return
        self._dirty_during_drag = False
        self.update_surface_iso()  # the heavy bit (iso-lines or whatever you draw)

    @staticmethod
    def _snap_value(v: float, anchor: float, step: float) -> float:
        return anchor + round((v - anchor) / step) * step

    # ----------------- Iso-lines (optional) -----------------

    def clear_iso(self) -> None:
        for it in self._iso_items:
            self.scene.removeItem(it)
        self._iso_items.clear()

    def add_polyline(self, pts: np.ndarray, z: float = 0.0) -> None:
        if pts.size == 0:
            return
        pts = np.asarray(pts, dtype=float)
        path = QtGui.QPainterPath(QtCore.QPointF(pts[0,0], pts[0,1]))
        for x, y in pts[1:]:
            path.lineTo(float(x), float(y))
        item = QtWidgets.QGraphicsPathItem(path)
        item.setPen(self._iso_pen)
        item.setZValue(z)
        self.scene.addItem(item)
        self._iso_items.append(item)

    def update_surface_iso(self, n_u: int = 25, n_v: int = 25) -> None:
        if self.state.base is None:
            return
        self.clear_iso()
        J, K = self.state.base.knot_vectors[0], self.state.base.knot_vectors[1]
        u_knots = np.unique(np.asarray(J, dtype=float))
        v_knots = np.unique(np.asarray(K, dtype=float))
        v_min, v_max = float(K[0]), float(K[-1])
        u_min, u_max = float(J[0]), float(J[-1])
        for u in u_knots:
            vs = np.linspace(v_min, v_max, n_v)
            params = np.column_stack([np.full_like(vs, u), vs])
            pts = np.asarray(self.state.base.evaluate(params))[:, :2]
            self.add_polyline(pts, z=-1)
        for v in v_knots:
            us = np.linspace(u_min, u_max, n_u)
            params = np.column_stack([us, np.full_like(us, v)])
            pts = np.asarray(self.state.base.evaluate(params))[:, :2]
            self.add_polyline(pts, z=-1)
        self.controlPointsChanged.emit()

class _ControlPointItem(QtCore.QObject, QtWidgets.QGraphicsEllipseItem):
    """Handle with constant on-screen radius; emits drag start/finish + position changes."""
    positionChanged = QtCore.Signal(int, QtCore.QPointF)
    dragStarted = QtCore.Signal(int)
    dragFinished = QtCore.Signal(int)

    def __init__(
        self,
        index: int,
        center: QtCore.QPointF,
        *,
        radius_px: float,
        normal_brush: QtGui.QBrush,
        selected_brush: QtGui.QBrush,
    ) -> None:
        QtCore.QObject.__init__(self)
        r = float(radius_px)
        QtWidgets.QGraphicsEllipseItem.__init__(self, -r, -r, 2*r, 2*r)

        self.index = index
        self._normal_brush = normal_brush
        self._selected_brush = selected_brush

        self.setFlags(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        )
        self.setAcceptHoverEvents(True)
        self.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self.setBrush(self._normal_brush)
        self.setPos(center)

    def set_center(self, p: QtCore.QPointF) -> None:
        self.setPos(p)

    # cursor + drag notifications
    def hoverEnterEvent(self, e: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        super().hoverEnterEvent(e)

    def mousePressEvent(self, e: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        self.dragStarted.emit(self.index)
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.dragFinished.emit(self.index)
        super().mouseReleaseEvent(e)

    def itemChange(self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.setBrush(self._selected_brush if self.isSelected() else self._normal_brush)
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.positionChanged.emit(self.index, self.scenePos())
        return super().itemChange(change, value)
