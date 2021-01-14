import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_absolute_deviation as sigmaMad
from scipy import stats as scipyStats

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

from .plotUtils import parsePlotInfo, addPlotInfo


class ColorColorFitPlotTaskConnections(pipeBase.PipelineTaskConnections,
                                       dimensions=("tract", "skymap"),
                                       defaultTemplates={"inputCoaddName": "deep",
                                                         "plotName": "deltaCoords"}):

    catPlot = pipeBase.connectionTypes.Input(doc="The tract wide catalog to make plots from.",
                                             storageClass="DataFrame",
                                             name="qaTable_tract",
                                             dimensions=("tract", "skymap"))

    colorColorFitPlot = pipeBase.connectionTypes.Output(doc="A scatter plot with histograms for both axes.",
                                                        storageClass="Plot",
                                                        name="colorColorFitPlot_{plotName}",
                                                        dimensions=("tract", "skymap"))

    skymap = pipeBase.connectionTypes.Input(doc="The skymap for the tract",
                                            storageClass="SkyMap",
                                            name="{inputCoaddName}Coadd_skyMap",
                                            dimensions=("skymap",))

    fitParams = pipeBase.connectionTypes.Input(doc="The parameters from the fit to the stellar locus.",
                                               storageClass="StructuredDataDict",
                                               name="stellarLocusParams_{plotName}",
                                               dimensions=("tract", "skymap"))


class ColorColorFitPlotTaskConfig(pipeBase.PipelineTaskConfig,
                                  pipelineConnections=ColorColorFitPlotTaskConnections):

    xColName = pexConfig.Field(
        doc="The column name for the values to be plotted on the x axis.",
        dtype=str,
        default="coord_ra",
    )

    yColName = pexConfig.Field(
        doc="The column name for the values to be plotted on the y axis.",
        dtype=str,
        default="coord_dec",
    )

    xLabel = pexConfig.Field(
        doc="The x axis label",
        dtype=str,
        default="Right Ascension (deg)",
    )

    yLabel = pexConfig.Field(
        doc="The y axis label",
        dtype=str,
        default="Declination (deg)",
    )

    name = pexConfig.Field(
        doc="The name of the subject of the plot.",
        dtype=str,
        default="yFit",
    )

    sourceTypeColName = pexConfig.Field(
        doc="The column to use for star - galaxy separation.",
        dtype=str,
        default="iExtendedness",
    )


class ColorColorFitPlotTask(pipeBase.PipelineTask):

    ConfigClass = ColorColorFitPlotTaskConfig
    _DefaultName = "ColorColorFitPlotTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class
        inputs = butlerQC.get(inputRefs)
        runName = inputRefs.catPlot.run
        dataId = butlerQC.quantum.dataId
        inputs["dataId"] = dataId
        inputs["runName"] = runName
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, catPlot, dataId, runName, skymap, fitParams):

        xLabel = self.config.xLabel
        yLabel = self.config.yLabel

        plotInfo = parsePlotInfo(dataId, runName)
        fig = self.colorColorFitPlot(catPlot, xLabel, yLabel, plotInfo, fitParams)

        return pipeBase.Struct(colorColorFitPlot=fig)

    def colorColorFitPlot(self, catPlot, xLabel, yLabel, plotInfo, fitParams):
        """Make stellar locus plots using pre fitted values.

        Parameters
        ----------
        catPlot : `pandas.core.frame.DataFrame`
            The catalog to plot the points from.
        xLabel : `str`
            The text to go on the xLabel of the plot.
        yLabel : `str`
            The text to go on the yLabel of the plot.
        plotInfo : `dict`
            A dictionary of information about the data being plotted with keys:
                ``"run"``
                    The output run for the plots (`str`).
                ``"skymap"``
                    The type of skymap used for the data (`str`).
                ``"filter"``
                    The filter used for this data (`str`).
                ``"tract"``
                    The tract that the data comes from (`str`).
        fitParams : `dict`
            The parameters of the fit to the stellar locus calculated
            elsewhere, they are used to plot the fit line on the
            figure.
                ``"bHW"``
                    The hardwired intercept to fall back on.
                ``"b_odr"``
                    The intercept calculated by the orthogonal distance
                    regression fitting.
                ``"mHW"``
                    The hardwired gradient to fall back on.
                ``"m_odr"``
                    The gradient calculated by the orthogonal distance
                    regression fitting.
                ``"magLim"``
                    The magnitude limit used in the fitting.
                ``"x1`"``
                    The x minimum of the box used in the fit.
                ``"x2"``
                    The x maximum of the box used in the fit.
                ``"y1"``
                    The y minimum of the box used in the fit.
                ``"y2"``
                    The y maximum of the box used in the fit.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The resulting figure.

        Notes
        -----
        Makes a color-color plot of `self.config.xColName` against
        `self.config.yColName`, these points are color coded by i band
        CModel magnitude. The stellar locus fits calculated from
        the calcStellarLocus task are then overplotted. The axis labels
        are given by `self.config.xLabel` and `self.config.yLabel`.
        The column given in `self.config.sourceTypeColName` is used for
        star galaxy separation. The distance of the points to the fit
        line is given in a histogram in the second panel.
        """

        self.log.info(("Plotting {}: the values of {} against {} on a color-color plot with the area "
                      "used for calculating the stellar locus fits marked.".format(
                       self.config.connections.plotName, self.config.xColName, self.config.yColName)))

        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.11, 0.3, 0.75])
        axHist = fig.add_axes([0.65, 0.11, 0.3, 0.75])

        # Cut the catalogue down to only valid sources
        catPlot = catPlot[catPlot["useForQAFlag"].values]

        # Only use sources brighter than 26th
        catPlot = catPlot[(catPlot["iCModelMag"] < fitParams["magLim"])]

        # Need to separate stars and galaxies
        stars = (catPlot[self.config.sourceTypeColName] == 0.0)

        # For stars
        xsStars = catPlot[self.config.xColName].values[stars]
        ysStars = catPlot[self.config.yColName].values[stars]

        # Points to use for the fit
        fitPoints = np.where((xsStars > fitParams["x1"]) & (xsStars < fitParams["x2"])
                             & (ysStars > fitParams["y1"]) & (ysStars < fitParams["y2"]))[0]
        ax.plot([fitParams["x1"], fitParams["x2"], fitParams["x2"], fitParams["x1"], fitParams["x1"]],
                [fitParams["y1"], fitParams["y1"], fitParams["y2"], fitParams["y2"], fitParams["y1"]], "k")

        bbox = dict(alpha=0.9, facecolor="white", edgecolor="none")
        ax.text(0.05, 0.95, "N Used: {}".format(len(fitPoints)), color="k", transform=ax.transAxes,
                fontsize=8, bbox=bbox)

        xyStars = np.vstack([xsStars, ysStars])
        starsKde = scipyStats.gaussian_kde(xyStars)
        zStars = starsKde(xyStars)

        ax.scatter(xsStars[~fitPoints], ysStars[~fitPoints], c=zStars[~fitPoints],
                   cmap="Greys_r", label="Stars", s=0.3)
        starFitPoints = ax.scatter(xsStars[fitPoints], ysStars[fitPoints], c=zStars[fitPoints], cmap="winter",
                                   label="Used for Fit", s=0.3)

        # Add colorbar
        starCbAx = fig.add_axes([0.43, 0.11, 0.04, 0.75])
        plt.colorbar(starFitPoints, cax=starCbAx)
        starCbAx.text(0.6, 0.5, "Number Density", color="k", rotation="vertical",
                      transform=starCbAx.transAxes, ha="center", va="center", fontsize=10)

        ax.set_xlabel(self.config.xLabel)
        ax.set_ylabel(self.config.yLabel)

        # Set useful axis limits
        starPercsX = np.nanpercentile(xsStars, [0.5, 100])
        starPercsY = np.nanpercentile(ysStars, [0.5, 100])
        ax.set_xlim(starPercsX[0], starPercsX[1])
        ax.set_ylim(starPercsY[0], starPercsY[1])

        # Fit a line to the points in the box
        xsFitLine = [fitParams["x1"], fitParams["x2"]]
        ysFitLineHW = [fitParams["mHW"]*xsFitLine[0] + fitParams["bHW"],
                       fitParams["mHW"]*xsFitLine[1] + fitParams["bHW"]]
        ax.plot(xsFitLine, ysFitLineHW, "w", lw=2)
        ax.plot(xsFitLine, ysFitLineHW, "g", lw=1, ls="--")

        ysFitLine = [fitParams["m_odr"]*xsFitLine[0] + fitParams["b_odr"],
                     fitParams["m_odr"]*xsFitLine[1] + fitParams["b_odr"]]
        ax.plot(xsFitLine, ysFitLine, "w", lw=2)
        ax.plot(xsFitLine, ysFitLine, "k", lw=1, ls="--")

        p1 = np.array([xsFitLine[0], ysFitLine[0]])
        p2 = np.array([xsFitLine[1], ysFitLine[1]])

        distsHW = []
        p1HW = np.array([xsFitLine[0], ysFitLineHW[0]])
        p2HW = np.array([xsFitLine[1], ysFitLineHW[1]])
        for point in zip(xsStars[fitPoints], ysStars[fitPoints]):
            point = np.array(point)
            distToLine = np.cross(p1HW - point, p2HW - point)/np.linalg.norm(p2HW - point)
            distsHW.append(distToLine)

        dists = []
        for point in zip(xsStars[fitPoints], ysStars[fitPoints]):
            point = np.array(point)
            distToLine = np.cross(p1 - point, p2 - point)/np.linalg.norm(p2 - point)
            dists.append(distToLine)

        # Add a histogram
        axHist.set_ylabel("Number")
        axHist.set_xlabel("Distance to Line Fit")
        axHist.hist(dists, bins=100, histtype="step")
        axHist.hist(distsHW, bins=100, histtype="step")
        medDists = np.median(dists)
        madDists = sigmaMad(dists)
        axHist.set_xlim(medDists - 4.0*madDists, medDists + 4.0*madDists)
        axHist.axvline(medDists, color="k", label="Median: {:0.3f}".format(medDists))
        axHist.axvline(medDists + madDists, color="k", ls="--", label="Sigma Mad: {:0.3f}".format(madDists))
        axHist.axvline(medDists - madDists, color="k", ls="--")
        axHist.legend(fontsize=8, loc="lower left")

        fig = addPlotInfo(plt.gcf(), plotInfo)

        return fig
