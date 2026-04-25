from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=8, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        #self.ax.set_xlim(-5, 5)
        #self.ax.set_ylim(-2, 4)
        self.ax.set_aspect('equal')
        super().__init__(fig)
        self.setParent(parent)