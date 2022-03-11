import numpy as np
import pandas as pd

import lsst.pipe.base as pipeBase

from .plotUtils import generateSummaryStatsVisit, parsePlotInfo
from .scatterPlot import ScatterPlotWithTwoHistsTask


class ScatterPlotVisitConnections(pipeBase.PipelineTaskConnections,
                                  dimensions=("visit", "skymap"),
                                  defaultTemplates={"plotName": "deltaCoords"}):

    catPlot = pipeBase.connectionTypes.Input(doc="The visit wide catalog to make plots from.",
                                             storageClass="DataFrame",
                                             name="sourceTable_visit",
                                             dimensions=("visit",),
                                             deferLoad=True)

    visitSummaryTable = pipeBase.connections.Input(doc="A summary table of the ccds in the visit",
                                                   storageClass="ExposureCatalog",
                                                   name="visitSummary",
                                                   dimensions=("visit",))

    scatterPlot = pipeBase.connectionTypes.Output(doc="A scatter plot with histograms for both axes.",
                                                  storageClass="Plot",
                                                  name="scatterPlotVisit_{plotName}",
                                                  dimensions=("visit", "skymap"))


class ScatterPlotVisitConfig(ScatterPlotWithTwoHistsTask.ConfigClass,
                             pipelineConnections=ScatterPlotVisitConnections):

    def setDefaults(self):
        super().setDefaults()
        self.axisActions.magAction.column = "psfFlux"
        self.axisActions.xAction.column = "psfFlux"
        self.highSnStatisticSelectorActions.statSelector.threshold = 100
        self.highSnStatisticSelectorActions.statSelector.bands = [""]
        self.lowSnStatisticSelectorActions.statSelector.threshold = 50
        self.lowSnStatisticSelectorActions.statSelector.bands = [""]
        self.sourceSelectorActions.sourceSelector.band = ""


class ScatterPlotVisitTask(ScatterPlotWithTwoHistsTask):
    """This task makes scatter plots with axis histograms
    for the given columns for visit level data. Stars are
    plotted in blue and galaxies in red."""

    ConfigClass = ScatterPlotVisitConfig
    _DefaultName = "scatterPlotVisitTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class
        columnNames = set(["detector"])
        for actionStruct in [self.config.axisActions, self.config.selectorActions,
                             self.config.highSnStatisticSelectorActions,
                             self.config.lowSnStatisticSelectorActions,
                             self.config.sourceSelectorActions]:
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
        inputs["plotName"] = localConnections.scatterPlot.name
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, catPlot, dataId, runName, tableName, visitSummaryTable, plotName):
        """Prep the catalogue and then make a scatterPlot of the given column.

        Parameters
        ----------
        catPlot : `pandas.core.frame.DataFrame`
            The catalog to plot the points from.
        dataId :
        `lsst.daf.butler.core.dimensions._coordinate._ExpandedTupleDataCoordinate`
            The dimensions that the plot is being made from.
        runName : `str`
            The name of the collection that the plot is written out to.
        tableName : `str`
            The type of table used to make the plot.

        Returns
        -------
        `pipeBase.Struct` containing:
            scatterPlot : `matplotlib.figure.Figure`
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
        in each patch for the column in self.zColName.

        `scatterPlotWithTwoHists` which makes a scatter plot of the points with
        a histogram of each axis.
        """

        # Apply the selectors to narrow down the sources to use
        mask = np.ones(len(catPlot), dtype=bool)
        for selector in self.config.selectorActions:
            mask &= selector(catPlot)
        catPlot = catPlot[mask]

        columns = {self.config.axisLabels["x"]: self.config.axisActions.xAction(catPlot),
                   self.config.axisLabels["y"]: self.config.axisActions.yAction(catPlot),
                   self.config.axisLabels["mag"]: self.config.axisActions.magAction(catPlot),
                   "detector": catPlot["detector"]}
        for actionStruct in [self.config.highSnStatisticSelectorActions,
                             self.config.lowSnStatisticSelectorActions,
                             self.config.sourceSelectorActions]:
            for action in actionStruct:
                for col in action.columns:
                    columns.update({col: catPlot[col]})
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
        useForStats = np.zeros(len(plotDf))
        lowSnMask = np.ones(len(plotDf), dtype=bool)
        for selector in self.config.lowSnStatisticSelectorActions:
            lowSnMask &= selector(plotDf)
        useForStats[lowSnMask] = 2

        highSnMask = np.ones(len(plotDf), dtype=bool)
        for selector in self.config.highSnStatisticSelectorActions:
            highSnMask &= selector(plotDf)
        useForStats[highSnMask] = 1
        plotDf.loc[:, "useForStats"] = useForStats

        # Get the S/N cut used
        try:
            SN = self.config.selectorActions.SnSelector.threshold
        except AttributeError:
            SN = "N/A"

        # Get useful information about the plot
        plotInfo = parsePlotInfo(dataId, runName, tableName, dataId["band"], plotName, SN)
        # Calculate the corners of the patches and some associated stats
        sumStats = generateSummaryStatsVisit(plotDf, self.config.axisLabels["y"], visitSummaryTable,
                                             plotInfo)
        # Make the plot
        fig = self.scatterPlotWithTwoHists(plotDf, plotInfo, sumStats)

        return pipeBase.Struct(scatterPlot=fig)
