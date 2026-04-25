# zoom_view.py
from PySide6 import QtWidgets, QtGui, QtCore
import math

class ZoomPanGraphicsView(QtWidgets.QGraphicsView):
    """
    Smooth, trackpad-friendly zoom under cursor.
    Middle mouse = pan. Keys: +/- (zoom), 0 (reset), F (fit-to-scene).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Rendering & anchors
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)

        # Pan state
        self._panning = False
        self._pan_last = QtCore.QPoint()

        # Zoom config
        self._min_scale = 0.02           # allow pretty deep zoom-out
        self._max_scale = 1e6            # practically unlimited zoom-in
        self._wheel_steps_per_2x = 6.0   # 6 notches doubles size
        self._trackpad_px_per_2x = 800   # 800px scroll doubles size

    # ---------- Coordinate system helper (optional Cartesian flip) ----------
    def _cartesian_view(self):
        sc = self.scene()
        if not sc:
            return
        h = sc.sceneRect().height()
        t = QtGui.QTransform()
        t.translate(0, h)
        t.scale(1, -1)
        self.setTransform(t)
        self._enforce_equal_aspect()

    def _enforce_equal_aspect(self) -> None:
        t = self.transform()

        # effective scales on each axis (robust to any rotation/shear)
        sx = math.hypot(t.m11(), t.m12())
        sy = math.hypot(t.m21(), t.m22())
        # choose a single uniform scale (you can also pick sx or sy)
        s = (sx + sy) * 0.5

        # keep current translation in device coords
        tx, ty = t.m31(), t.m32()

        # preserve your Cartesian flip if present (negative y scale)
        y_flipped = (t.m11() * t.m22() - t.m12() * t.m21()) < 0 and (t.m12() == 0 and t.m21() == 0)

        # rebuild as pure (uniform) scale + translation, no shear/rotation
        new = QtGui.QTransform(
            s, 0, 0,
            0, -s if y_flipped else s, 0,
            tx, ty, 1.0
        )
        super().setTransform(new, False)

    def setTransform(self, transform: QtGui.QTransform, combine: bool = False) -> None:
        super().setTransform(transform, combine)
        self._enforce_equal_aspect()


    # ---------- Fit & reset that respect the Cartesian transform ----------
    def fit_to_scene(self, padding: float = 20.0) -> None:
        sc = self.scene()
        if not sc:
            return
        rect = sc.itemsBoundingRect().adjusted(-padding, -padding, padding, padding)
        if rect.isEmpty():
            return
        super().resetTransform()  # ensure no compounding/jump
        super().fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self._cartesian_view()    # reapply flip

    def resetTransform(self) -> None:
        super().resetTransform()
        self._cartesian_view()    # reapply flip

    # ---------- Zoom ----------
    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self.scene() is None:
            return

        pix = event.pixelDelta().y()
        ang = event.angleDelta().y()

        if pix:
            factor = 2.0 ** (pix / self._trackpad_px_per_2x)
        elif ang:
            steps = ang / 120.0
            factor = 2.0 ** (steps / self._wheel_steps_per_2x)
        else:
            return

        self._apply_zoom(factor)
        event.accept()
        self._enforce_equal_aspect()

    def _apply_zoom(self, factor: float) -> None:
        """Apply zoom factor without snapping if clamps are exceeded."""
        current = self.transform().m11()
        target  = current * factor

        # If trying to zoom in but we're already beyond max, ignore (no snap).
        if factor > 1.0 and current >= self._max_scale:
            return
        # If trying to zoom out but we're already below min, ignore.
        if factor < 1.0 and current <= self._min_scale:
            return

        # Soft clamp toward limits (only if requested target crosses them)
        if target > self._max_scale and factor > 1.0:
            target = self._max_scale
        if target < self._min_scale and factor < 1.0:
            target = self._min_scale

        # Apply only the incremental delta
        to_apply = target / current
        if to_apply != 1.0:
            self.scale(to_apply, to_apply)
        self._enforce_equal_aspect()

    # ---------- Pan (middle mouse via scrollbars) ----------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_last = event.pos()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._panning:
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            self._pan_by(delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def _pan_by(self, delta: QtCore.QPoint) -> None:
        # Use scrollbars for robust panning, independent of transforms
        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        hbar.setValue(hbar.value() - delta.x())
        vbar.setValue(vbar.value() - delta.y())

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        if self._panning:
            self._panning = False
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        super().leaveEvent(event)

    # ---------- Keyboard ----------
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key in (QtCore.Qt.Key.Key_Plus, QtCore.Qt.Key.Key_Equal):
            self._apply_zoom(2.0 ** (1.0 / self._wheel_steps_per_2x))
            return
        if key == QtCore.Qt.Key.Key_Minus:
            self._apply_zoom(2.0 ** (-1.0 / self._wheel_steps_per_2x))
            return
        if key == QtCore.Qt.Key.Key_0:
            self.resetTransform()
            return
        if key in (QtCore.Qt.Key.Key_F, QtCore.Qt.Key.Key_Home):
            self.fit_to_scene()
            return
        super().keyPressEvent(event)

    # ---------- Replace helper ----------
    @staticmethod
    def replace(existing_view: QtWidgets.QGraphicsView) -> "ZoomPanGraphicsView":
        parent = existing_view.parentWidget()
        new = ZoomPanGraphicsView(parent)
        new.setObjectName(existing_view.objectName())
        new.setScene(existing_view.scene())
        if parent and parent.layout():
            layout = parent.layout()
            i = layout.indexOf(existing_view)
            layout.insertWidget(i, new)
        else:
            new.setGeometry(existing_view.geometry())
        existing_view.setParent(None)
        existing_view.deleteLater()
        return new
