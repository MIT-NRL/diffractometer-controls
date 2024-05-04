import matplotlib.figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QSizePolicy, QTabWidget, QVBoxLayout, QWidget
from bluesky.callbacks.core import make_callback_safe, make_class_safe
from bluesky.callbacks import LiveTable, LivePlot
import numpy as np


def _initialize_matplotlib():
    "Set backend to Qt5Agg and import pyplot."
    import matplotlib

    matplotlib.use("Qt5Agg")  # must set before importing matplotlib.pyplot
    import matplotlib.pyplot  # noqa



class QtFigure(QWidget):
    """
    A Qt view for a Figure model. This always contains one Figure.
    """

    def __init__(self, parent=None):
        _initialize_matplotlib()
        super().__init__(parent)
        # self.model = model
        self.figure = matplotlib.figure.Figure()
        self.figure.set_tight_layout(True)
        # TODO Let Figure give different options to subplots here,
        # but verify that number of axes created matches the number of axes
        # specified.
        # self.axes_list = list(self.figure.subplots(len(model.axes), squeeze=False).ravel())

        # self.figure.suptitle(model.title)
        self.axes = self.figure.add_subplot(111)
        self.axes.grid(alpha=0.25)
        self.axes.tick_params(direction='in', which='both')
        canvas = FigureCanvas(self.figure)
        canvas.setMinimumWidth(640)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.updateGeometry()
        canvas.setParent(self)
        toolbar = NavigationToolbar(canvas, parent=self)

        layout = QVBoxLayout()
        layout.addWidget(canvas)
        layout.addWidget(toolbar)
        self.setLayout(layout)
        self.resize(self.sizeHint())

        # model.events.title.connect(self._on_title_changed)
        # The Figure model does not currently allow axes to be added or
        # removed, so we do not need to handle changes in model.axes.

    def sizeHint(self):
        size_hint = super().sizeHint()
        size_hint.setWidth(700)
        size_hint.setHeight(500)
        return size_hint

    # @property
    # def axes(self):
    #     "Read-only access to the mapping Axes UUID -> MatplotlibAxes"
    #     return self._axes

    def _on_title_changed(self, event):
        self.figure.suptitle(event.value)
        self._redraw()

    def _redraw(self):
        "Redraw the canvas."
        # Schedule matplotlib to redraw the canvas at the next opportunity, in
        # a threadsafe fashion.
        self.figure.canvas.draw_idle()

    def close_figure(self):
        self.figure.canvas.close()


@make_class_safe
class NewLivePlot(LivePlot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def event(self, doc):
        # print('event data',doc)
        "Unpack data from the event and call self.update()."
        # This outer try/except block is needed because multiple event
        # streams will be emitted by the RunEngine and not all event
        # streams will have the keys we want.
        # try:
        #     # This inner try/except block handles seq_num and time, which could
        #     # be keys in the data or accessing the standard entries in every
        #     # event.
        #     try:
        #         new_x = doc["data"][self.x]
        #     except KeyError:
        #         if self.x in ("time", "seq_num"):
        #             new_x = doc[self.x]
        #         else:
        #             raise
        #     new_y = doc["data"][self.y]
        #     print('new_y',new_y)
        # except KeyError:
        #     # wrong event stream, skip it
        #     return
        new_y = doc["data"][self.y]
        new_x = np.linspace(-150,150,300)
        
        # print(len(new_x))
        # print(new_x.shape,new_y.shape)

        # Special-case 'time' to plot against against experiment epoch, not
        # UNIX epoch.
        if self.x == "time" and self._epoch == "run":
            new_x -= self._epoch_offset

        self.update_caches(new_x, new_y)
        self.update_plot()
        super().event(doc)

    def update_caches(self, x, y):
        self.x_data = x
        self.y_data = y