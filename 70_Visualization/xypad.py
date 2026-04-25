# Fully chatgpt designed
from PySide6.QtCore import Qt, QPointF, QRectF, QSize, Signal
from PySide6.QtGui import QPainter, QPen, QBrush, QFontMetrics
from PySide6.QtWidgets import QWidget
from math import ceil, floor, isfinite

class XYPad(QWidget):
    pointChanged = Signal(float, float)   # x, y
    xChanged = Signal(float)
    yChanged = Signal(float)

    def __init__(self, parent=None, x_range=None, y_range=None, step=(0.1, 0.1)):
        super().__init__(parent)
        if x_range is None: x_range = (-2.0, 2.0)
        if y_range is None: y_range = (-1.0, 3.0)
        self._xmin, self._xmax = map(float, x_range)
        self._ymin, self._ymax = map(float, y_range)
        self._step_x, self._step_y = float(step[0]), float(step[1])
        self._x = 0.0
        self._y = 0.0
        self._margin = 28  # room for labels
        self._handle_r = 6
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # -------- Public API -----------------------------------------------------
    def sizeHint(self) -> QSize:
        return QSize(220, 220)

    def ranges(self):
        return (self._xmin, self._xmax), (self._ymin, self._ymax)

    def setRanges(self, x_range, y_range):
        self._xmin, self._xmax = map(float, x_range)
        self._ymin, self._ymax = map(float, y_range)
        self.setPoint(self._x, self._y)  # re-clamp & redraw

    def setStep(self, step_x=None, step_y=None):
        if step_x is not None: self._step_x = float(step_x)
        if step_y is not None: self._step_y = float(step_y)
        self.setPoint(self._x, self._y)

    def value(self):
        return self._x, self._y

    def setPoint(self, x, y):
        # clamp
        x = max(self._xmin, min(self._xmax, x))
        y = max(self._ymin, min(self._ymax, y))
        # snap to step (anchored at vmin)
        x = self._snap(x, self._xmin, self._step_x)
        y = self._snap(y, self._ymin, self._step_y)
        changed = (x != self._x) or (y != self._y)
        self._x, self._y = x, y
        if changed:
            self.pointChanged.emit(self._x, self._y)
            self.xChanged.emit(self._x)
            self.yChanged.emit(self._y)
            self.update()

    # -------- Events ---------------------------------------------------------
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self._padRect()

        # background
        p.fillRect(self.rect(), self.palette().window())
        p.fillRect(rect, self.palette().base())

        # minor grid (optional): based on step
        if self._step_x > 0 or self._step_y > 0:
            p.setPen(QPen(self.palette().mid().color(), 1, Qt.SolidLine))
            if self._step_x > 0:
                x = ceil(self._xmin / self._step_x) * self._step_x
                while x <= self._xmax + 1e-12:
                    px = self._valueToPos(x, self._ymin).x()
                    p.drawLine(int(px), rect.top(), int(px), rect.bottom())
                    x += self._step_x
            if self._step_y > 0:
                y = ceil(self._ymin / self._step_y) * self._step_y
                while y <= self._ymax + 1e-12:
                    py = self._valueToPos(self._xmin, y).y()
                    p.drawLine(rect.left(), int(py), rect.right(), int(py))
                    y += self._step_y

        # axes at zero (if visible)
        p.setPen(QPen(self.palette().text().color(), 1))
        if self._xmin < 0 < self._xmax:
            x0 = self._valueToPos(0, self._ymin).x()
            p.drawLine(int(x0), rect.top(), int(x0), rect.bottom())
        if self._ymin < 0 < self._ymax:
            y0 = self._valueToPos(self._xmin, 0).y()
            p.drawLine(rect.left(), int(y0), rect.right(), int(y0))

        # integer major ticks & labels only
        fm = QFontMetrics(p.font())
        label_pad = 4
        p.setPen(QPen(self.palette().text().color(), 1))

        xi = ceil(self._xmin)
        while xi <= floor(self._xmax):
            px = self._valueToPos(xi, self._ymin).x()
            p.drawLine(int(px), rect.top(), int(px), rect.bottom())
            txt = str(int(xi))
            tw = fm.horizontalAdvance(txt)
            p.drawText(int(px - tw / 2), int(rect.bottom() + fm.ascent() + label_pad), txt)
            xi += 1

        yi = ceil(self._ymin)
        while yi <= floor(self._ymax):
            py = self._valueToPos(self._xmin, yi).y()
            p.drawLine(rect.left(), int(py), rect.right(), int(py))
            txt = str(int(yi))
            tw = fm.horizontalAdvance(txt)
            p.drawText(int(rect.left() - tw - label_pad), int(py + fm.ascent() / 2), txt)
            yi += 1

        # handle on top
        pos = self._valueToPos(self._x, self._y)
        p.setBrush(QBrush(self.palette().highlight().color()))
        p.setPen(QPen(self.palette().highlightedText().color(), 1))
        p.drawEllipse(pos, self._handle_r, self._handle_r)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._updateFromPos(e.position())
            self.setFocus()

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._updateFromPos(e.position())

    def keyPressEvent(self, e):
        # Nudge by step (not % range). If step <= 0, fall back to 1% of range.
        dx = self._step_x if self._step_x > 0 else 0.01 * (self._xmax - self._xmin)
        dy = self._step_y if self._step_y > 0 else 0.01 * (self._ymax - self._ymin)
        if e.key() == Qt.Key_Left:   self.setPoint(self._x - dx, self._y)
        elif e.key() == Qt.Key_Right: self.setPoint(self._x + dx, self._y)
        elif e.key() == Qt.Key_Down:  self.setPoint(self._x, self._y - dy)
        elif e.key() == Qt.Key_Up:    self.setPoint(self._x, self._y + dy)
        else:
            super().keyPressEvent(e)

    # -------- Helpers --------------------------------------------------------
    @staticmethod
    def _snap(v, vmin, step):
        if not isfinite(step) or step <= 0:
            return v
        k = round((v - vmin) / step)
        snapped = vmin + k * step
        # round to decimals implied by step to suppress float noise
        s = f"{step:.12g}"
        decimals = len(s.split(".")[1]) if "." in s else 0
        return round(snapped, decimals)

    def _padRect(self) -> QRectF:
        r = self.rect().adjusted(self._margin, self._margin, -self._margin, -self._margin)
        s = min(r.width(), r.height())
        cx, cy = r.center().x(), r.center().y()
        return QRectF(cx - s/2, cy - s/2, s, s)

    def _posToValue(self, pos: QPointF):
        rect = self._padRect()
        x_frac = (pos.x() - rect.left()) / max(1.0, rect.width())
        y_frac = (pos.y() - rect.top())  / max(1.0, rect.height())
        x_frac = max(0.0, min(1.0, x_frac))
        y_frac = max(0.0, min(1.0, y_frac))
        x = self._xmin + x_frac * (self._xmax - self._xmin)
        y = self._ymax - y_frac * (self._ymax - self._ymin)  # up is positive
        return x, y

    def _valueToPos(self, x, y) -> QPointF:
        rect = self._padRect()
        x_frac = (x - self._xmin) / (self._xmax - self._xmin or 1.0)
        y_frac = (self._ymax - y) / (self._ymax - self._ymin or 1.0)
        px = rect.left() + x_frac * rect.width()
        py = rect.top()  + y_frac * rect.height()
        return QPointF(px, py)

    def _updateFromPos(self, pos):
        x, y = self._posToValue(pos)
        self.setPoint(x, y)
