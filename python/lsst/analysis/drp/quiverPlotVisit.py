import lsst.pipe.base as pipeBase

from .skyPlotVisit import SkyPlotVisitTask, SkyPlotVisitTaskConnections
from .quiverPlot import QuiverPlotTask

__all__ = ["QuiverPlotVisitTaskConnections", "QuiverPlotVisitTaskConfig", "QuiverPlotVisitTask"]


class QuiverPlotVisitTaskConnections(SkyPlotVisitTaskConnections):

    skyPlot = pipeBase.connectionTypes.Output(doc="A plot showing the on-sky distribution of a value.",
                                              storageClass="Plot",
                                              name="quiverPlotVisit_{plotName}",
                                              dimensions=("visit",))


class QuiverPlotVisitTaskConfig(SkyPlotVisitTask.ConfigClass, QuiverPlotTask.ConfigClass,
                                pipelineConnections=QuiverPlotVisitTaskConnections):
    # The defaults from the two inherited configs are all that is required.
    pass


class QuiverPlotVisitTask(QuiverPlotTask, SkyPlotVisitTask):
    """A task to make a plot that shows the on sky distribution of spin-2
    quantities, usually ellipticities. These plots are useful to visualize
    the spatial pattern to the plotted variable.
    Either in terms of RA/Dec or corresponding to the overplotted
    ccd outlines.
    """

    ConfigClass = QuiverPlotVisitTaskConfig
    _DefaultName = "quiverPlotVisitTask"
