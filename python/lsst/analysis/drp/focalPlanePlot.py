import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic_2d
import matplotlib.patheffects as pathEffects
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import lsst.afw.cameraGeom as cameraGeom
from lsst.pipe.tasks.configurableActions import ConfigurableActionStructField
from lsst.skymap import BaseSkyMap

from . import dataSelectors
from .plotUtils import parsePlotInfo, addPlotInfo


class FocalPlanePlotConnections(pipeBase.PipelineTaskConnections,
                                dimensions=("tract", "skymap", "instrument"),
                                defaultTemplates={"inputCoaddName": "deep", "plotName": "DIA_object_count",
                                                  "tableType": "forced"}):

    catPlot = pipeBase.connectionTypes.Input(doc="The tract-wide catalog to make plots from.",
                                             storageClass="DataFrame",
                                             name="diaSourceTable_tract",
                                             dimensions=("tract", "skymap"),
                                             deferLoad=True)

    skymap = pipeBase.connectionTypes.Input(doc="The skymap for the tract",
                                            storageClass="SkyMap",
                                            name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
                                            dimensions=("skymap",))

    camera = pipeBase.connectionTypes.Input(doc="The camera model",
                                            storageClass="Camera",
                                            name="camera",
                                            isCalibration=True,
                                            dimensions=('instrument',))

    focalPlanePlot = pipeBase.connectionTypes.Output(doc=("A plot showing the focal plane distribution of a"
                                                          "value."),
                                                     storageClass="Plot",
                                                     name="focalPlane_{plotName}",
                                                     dimensions=("tract", "skymap", "instrument"))


class FocalPlanePlotConfig(pipeBase.PipelineTaskConfig, pipelineConnections=FocalPlanePlotConnections):

    binsize = pexConfig.Field(
        doc="Bin size for 2D histogram (focal plane units)",
        default=2,
        dtype=float
    )

    statistic = pexConfig.Field(
        doc="Operation to perform in binned_statistic_2d",
        default='count',
        dtype=str
    )

    statisticColumn = pexConfig.Field(
        doc="Field to use for binned_statistic_2d. Not used if statistic operation is `count`",
        default='x',
        dtype=str
    )

    convertToFP = pexConfig.Field(
        doc="Convert coordinates to focal plane?",
        default=True,
        dtype=bool
    )

    selectorActions = ConfigurableActionStructField(
        doc="Which selectors to use to narrow down the data for QA plotting.",
        default={"flagSelector": dataSelectors.FlagSelector},
    )


class FocalPlanePlotTask(pipeBase.PipelineTask):

    ConfigClass = FocalPlanePlotConfig
    _DefaultName = "focalPlanePlotTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class

        inputs = butlerQC.get(inputRefs)

        columnNames = ['x', 'y']

        if self.config.convertToFP:
            columnNames.append('ccdVisitId')
        columnNames.append(self.config.statisticColumn)

        for action in self.config.selectorActions:
            for col in action.columns:
                columnNames.append(col)

        dataFrame = inputs["catPlot"].get(parameters={"columns": columnNames})
        inputs['catPlot'] = dataFrame

        instrumentDataId = butlerQC.registry.expandDataId(instrument=butlerQC.quantum.dataId["instrument"])
        packer = butlerQC.registry.dimensions.makePacker("visit_detector", instrumentDataId)

        localConnections = self.config.ConnectionsClass(config=self.config)
        plotInfo = parsePlotInfo(butlerQC.quantum.dataId,
                                 inputRefs.catPlot.datasetRef.run,
                                 localConnections.catPlot.name,
                                 "",
                                 localConnections.focalPlanePlot.name,
                                 "N/A")

        outputs = self.run(**inputs, packer=packer, plotInfo=plotInfo)
        butlerQC.put(outputs, outputRefs)

    def run(self, catPlot, skymap, camera, packer=None, plotInfo=None):

        mask = np.ones(len(catPlot), dtype=bool)
        for selector in self.config.selectorActions:
            mask &= selector(catPlot)
        catPlot = catPlot[mask].copy(deep=True)

        xCol = 'x'
        yCol = 'y'

        if self.config.convertToFP:
            ccdVisitIds = packer.unpack(catPlot['ccdVisitId'])
            detectorIds = ccdVisitIds["detector"]

            xCol = 'x_fp'
            yCol = 'y_fp'
            catPlot.loc[:, 'x_fp'] = np.zeros(len(catPlot))
            catPlot.loc[:, 'y_fp'] = np.zeros(len(catPlot))
            catPlot.loc[:, 'test'] = np.zeros(len(catPlot))
            for detectorId in detectorIds.unique():
                detIndex = detectorIds == detectorId
                detector_sources = catPlot[detIndex]
                points = detector_sources[["x", "y"]].to_numpy().T

                detector = camera[detectorId]
                map = detector.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE).getMapping()
                focalPlane_xy = map.applyForward(points)
                catPlot.loc[detIndex, 'x_fp'] = focalPlane_xy[0]
                catPlot.loc[detIndex, 'y_fp'] = focalPlane_xy[1]
                catPlot.loc[detIndex, 'test'] = detectorId

        binsx = np.arange(catPlot[xCol].min(), catPlot[yCol].max() + self.config.binsize,
                          self.config.binsize)
        binsy = np.arange(catPlot[xCol].min(), catPlot[yCol].max() + self.config.binsize,
                          self.config.binsize)

        statistic, x_edge, y_edge, binnumber = binned_statistic_2d(catPlot[xCol], catPlot[yCol],
                                                                   catPlot[self.config.statisticColumn],
                                                                   statistic=self.config.statistic,
                                                                   bins=[binsx, binsy])
        binExtent = [x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]]

        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        plot = ax.imshow(statistic, extent=binExtent)
        ax.set_xlabel(r'$x_{focal\:plane}$')
        ax.set_ylabel(r'$y_{focal\:plane}$')
        ax.tick_params(axis="x", labelrotation=25)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")

        cax = fig.add_axes([0.87 + 0.04, 0.11, 0.04, 0.77])
        plt.colorbar(plot, cax=cax, extend="both")
        colorBarLabel = "Counts"
        text = cax.text(0.5, 0.5, colorBarLabel, color="k", rotation="vertical", transform=cax.transAxes,
                        ha="center", va="center", fontsize=10)
        text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])
        cax.tick_params(labelsize=7)

        cax.yaxis.set_ticks_position("left")
        plt.draw()

        fig = plt.gcf()
        fig = addPlotInfo(fig, plotInfo)

        return pipeBase.Struct(focalPlanePlot=fig)
