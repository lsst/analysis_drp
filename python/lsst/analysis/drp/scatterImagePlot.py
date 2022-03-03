import pandas as pd
from lsst.skymap import BaseSkyMap

from lsst.pipe.tasks.configurableActions import ConfigurableActionStructField
from lsst.pipe.tasks.dataFrameActions import SingleColumnAction
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

# from . import dataSelectors as dataSelectors
# from .scatterPlot import ScatterPlotWithTwoHistsTask
from .plotUtils import generateSummaryStats, parsePlotInfo

from .imageAnalysis import maskPixelsPercentCalc


class ScatterPlotFromImageTaskConnections(pipeBase.PipelineTaskConnections,
                                          dimensions=("tract", "skymap"),
                                          defaultTemplates={"inputCoaddName": "deep",
                                                            "plotName": "deltaCoords"}):

    images = pipeBase.connectionTypes.Input(doc="The image to make plots from.",
                                            storageClass="ExposureF",
                                            name="{inputCoaddName}Coadd_calexp",
                                            dimensions=("tract", "patch", "band", "skymap"),
                                            multiple=True)

    skymap = pipeBase.connectionTypes.Input(doc="The skymap for the tract",
                                            storageClass="SkyMap",
                                            name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
                                            dimensions=("skymap",))

    scatterPlot = pipeBase.connectionTypes.Output(doc="A scatter plot with histograms for both axes.",
                                                  storageClass="Plot",
                                                  name="scatterTwoHistPlot_{plotName}",
                                                  dimensions=("tract", "skymap", "band"))


class ScatterPlotFromImageTaskConfig(pipeBase.PipelineTaskConfig,
                                     pipelineConnections=ScatterPlotFromImageTaskConnections):

    axisLabels = pexConfig.DictField(
        doc="Name of the dataframe columns to plot, will be used as the axis label: {'x':, 'y':, 'mag':}"
            "The mag column is used to decide which points to include in the printed statistics.",
        keytype=str,
        itemtype=str
    )

    axisActions = ConfigurableActionStructField(
        doc="",
        default={"yAction": SingleColumnAction},
    )

    nBins = pexConfig.Field(
        doc="Number of bins to put on the x axis.",
        default=40.0,
        dtype=float,
    )

    def setDefaults(self):
        super().setDefaults()


class ScatterPlotFromImageTask(pipeBase.PipelineTask):

    ConfigClass = ScatterPlotFromImageTaskConfig
    _DefaultName = "scatterPlotFromImageTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class
        inputs = butlerQC.get(inputRefs)

        imDict = {ref.dataId["patch"]: image for (ref, image) in zip(inputRefs.images, inputs["images"])}
        dataId = butlerQC.quantum.dataId
        inputs["images"] = imDict
        inputs["dataId"] = dataId
        inputs["band"] = inputRefs.images[0].dataId["band"]
        inputs["runName"] = inputRefs.images[0].run
        localConnections = self.config.ConnectionsClass(config=self.config)
        inputs["tableName"] = localConnections.images.name
        inputs["plotName"] = localConnections.scatterPlot.name
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, images, dataId, runName, skymap, tableName, band, plotName):
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
        skymap : `lsst.skymap`
            The skymap used to define the patch boundaries.
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

        # Make a small DF with columns patch and number of mask pixels.
        patches = []
        yVals = []

        for patch in images.keys():
            patches.append(patch)
            yVals.append(maskPixelsPercentCalc(images[patch], 'NO_DATA'))

        columns = {"patchId": patches, self.config.axisLabels["y"]: yVals}
        plotDf = pd.DataFrame(columns)

        # Get useful information about the plot
        plotInfo = parsePlotInfo(dataId, runName, tableName, band, plotName, "N/A")
        # Calculate the corners of the patches and some associated stats
        sumStats = generateSummaryStats(plotDf, self.config.axisLabels["y"], skymap, plotInfo)
        # Make the plot
        fig = self.scatterPlotWithTwoHists(plotDf, plotInfo, sumStats)

        return pipeBase.Struct(scatterPlot=fig)
