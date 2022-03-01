import numpy as np
import lsst.pipe.base as pipeBase

from .plotUtils import generateSummaryStatsVisit, parsePlotInfo
from .skyPlot import SkyPlotTask
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["SkyPlotVisitTaskConfig", "SkyPlotVisitTask"]


class SkyPlotVisitTaskConnections(pipeBase.PipelineTaskConnections, dimensions=("visit", "skymap"),
                                  defaultTemplates={"plotName": "deltaCoords"}):

    catPlot = pipeBase.connectionTypes.Input(doc="The visit-wide catalog to make plots from.",
                                             storageClass="DataFrame",
                                             name="sourceTable_visit",
                                             dimensions=("visit",),
                                             deferLoad=True)

    visitSummaryTable = pipeBase.connectionTypes.Input(doc="A summary table of the ccds in the visit",
                                                       storageClass="ExposureCatalog",
                                                       name="visitSummary",
                                                       dimensions=("visit",))

    skyPlot = pipeBase.connectionTypes.Output(doc="A plot showing the on-sky distribution of a value.",
                                              storageClass="Plot",
                                              name="skyPlotVisit_{plotName}",
                                              dimensions=("visit",))


class SkyPlotVisitTaskConfig(SkyPlotTask.ConfigClass, pipelineConnections=SkyPlotVisitTaskConnections):

    def setDefaults(self):
        super().setDefaults()
        self.axisActions.xAction.column = "coord_ra"
        self.axisActions.xAction.inRadians = False
        self.axisActions.yAction.column = "coord_dec"
        self.axisActions.yAction.inRadians = False
        self.sourceSelectorActions.sourceSelector.band = ""
        self.statisticSelectorActions.statSelector.bands = [""]
        self.statisticSelectorActions.statSelector.threshold = 100


class SkyPlotVisitTask(SkyPlotTask):
    """A task to make a plot that shows the on sky distribution of a
    given column. Stars are plotted using a blue color map, galaxies
    with a red one and when all sources are shown it is plotted
    using a purple colormap. These plots are useful to study whether
    or not there is a spatial distribution to the plotted variable.
    Either in terms of RA/Dec or corresponding to the overplotted
    ccd outlines.
    """

    ConfigClass = SkyPlotVisitTaskConfig
    _DefaultName = "skyPlotVisitTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class

        columnNames = set(["detector"])
        for actionStruct in [self.config.axisActions, self.config.selectorActions,
                             self.config.statisticSelectorActions, self.config.sourceSelectorActions]:
            for action in actionStruct:
                for col in action.columns:
                    columnNames.add(col)

        inputs = butlerQC.get(inputRefs)
        dataFrame = inputs["catPlot"].get(parameters={"columns": columnNames})
        inputs['catPlot'] = dataFrame
        dataId = butlerQC.quantum.dataId
        inputs["dataId"] = dataId
        inputs["runName"] = inputRefs.catPlot.datasetRef.run
        localConnections = self.config.ConnectionsClass(config=self.config)
        inputs["tableName"] = localConnections.catPlot.name
        inputs["plotName"] = localConnections.skyPlot.name
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)
        plt.close()

    def run(self, catPlot, dataId, runName, tableName, visitSummaryTable, plotName):
        """Prep the catalogue and then make a skyPlot of the given column.

        Parameters
        ----------
        catPlot : `pandas.core.frame.DataFrame`
            The catalog to plot the points from.
        dataId :
        `lsst.daf.butler.core.dimensions._coordinate._ExpandedTupleDataCoordinate`
            The dimensions that the plot is being made from.
        runName : `str`
            The name of the collection that the plot is written out to.
        skymap : `lsst.skymap`
            The skymap used to define the patch boundaries.
        tableName : `str`
            The type of table used to make the plot.

        Returns
        -------
        `pipeBase.Struct` containing:
            skyPlot : `matplotlib.figure.Figure`
                The resulting figure.

        Notes
        -----
        The catalogue is first narrowed down using the selectors specified in
        `self.config.selectorActions`.
        If the column names are 'Functor' then the functors specified in
        `self.config.axisFunctors` are used to calculate the required values.
        After this the following functions are run:

        `parsePlotInfo` which uses the dataId, runName and tableName to add
        useful information to the plot.

        `generateSummaryStats` which parses the skymap to give the corners of
        the patches for later plotting and calculates some basic statistics
        in each patch for the column in self.config.axisActions['zAction'].

        `SkyPlot` which makes the plot of the sky distribution of
        `self.config.axisActions['zAction']`.

        """

        # Apply the selectors to narrow down the sources to use
        mask = np.ones(len(catPlot), dtype=bool)
        for selector in self.config.selectorActions:
            mask &= selector(catPlot)
        catPlot = catPlot[mask]

        columns = {self.config.axisLabels["x"]: self.config.axisActions.xAction(catPlot),
                   self.config.axisLabels["y"]: self.config.axisActions.yAction(catPlot),
                   self.config.axisLabels["z"]: self.config.axisActions.zAction(catPlot),
                   "detector": catPlot["detector"]}
        plotDf = pd.DataFrame(columns)

        sourceTypes = np.zeros(len(plotDf))
        for selector in self.config.sourceSelectorActions:
            # The source selectors return 1 for a star and 2 for a galaxy
            # rather than a mask this allows the information about which
            # type of sources are being plotted to be propagated
            sourceTypes += selector(catPlot)
        if list(self.config.sourceSelectorActions) == []:
            sourceTypes = [10]*len(plotDf)
        plotDf.loc[:, "sourceType"] = sourceTypes

        # Decide which points to use for stats calculation
        plotDf.loc[:, "useForStats"] = self.config.statisticSelectorActions.statSelector(catPlot)

        # Check the columns have finite values
        mask = np.ones(len(catPlot), dtype=bool)
        for col in plotDf.columns:
            mask &= np.isfinite(plotDf[col])
        plotDf = plotDf[mask]

        # Get the S/N cut
        try:
            SN = self.config.selectorActions.SnSelector.threshold
        except AttributeError:
            SN = "N/A"

        # Get useful information about the plot
        plotInfo = parsePlotInfo(dataId, runName, tableName, dataId["band"], plotName, SN)
        # Calculate the corners of the patches and some associated stats
        sumStats = generateSummaryStatsVisit(plotDf, self.config.axisLabels["z"], visitSummaryTable,
                                             plotInfo)
        # Make the plot
        fig = self.skyPlot(plotDf, plotInfo, sumStats)

        return pipeBase.Struct(skyPlot=fig)
