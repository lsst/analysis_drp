import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_absolute_deviation as sigmaMad
from matplotlib import patheffects
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pathEffects
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from .configStructField import ConfigStructField
from lsst.skymap import BaseSkyMap

from .dataSelectors import MainFlagSelector, LowSnSelector
from .calcFunctors import RadToDeg, SNCalculator
from .plotUtils import generateSummaryStats, parsePlotInfo, addPlotInfo

# Changing this because it keeps throwing warnings despite using the
# recommended syntax
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


class SkyPlotTaskConnections(pipeBase.PipelineTaskConnections, dimensions=("tract", "skymap"),
                             defaultTemplates={"inputCoaddName": "deep", "plotName": "deltaCoords",
                                               "tableType": "forced"}):

    catPlot = pipeBase.connectionTypes.Input(doc="The tract wide catalog to make plots from.",
                                             storageClass="DataFrame",
                                             name="qaTractTable_{tableType}",
                                             dimensions=("tract", "skymap", "band"))

    skymap = pipeBase.connectionTypes.Input(doc="The skymap for the tract",
                                            storageClass="SkyMap",
                                            name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
                                            dimensions=("skymap",))

    skyPlot = pipeBase.connectionTypes.Output(doc="A plot showing the on sky distribution of a value.",
                                              storageClass="Plot",
                                              name="skyPlot_{plotName}",
                                              dimensions=("tract", "skymap"))


class SkyPlotTaskConfig(pipeBase.PipelineTaskConfig, pipelineConnections=SkyPlotTaskConnections):

    # These can stop being functors as defaults if the catalogue
    # moves natively to degrees
    xColName = pexConfig.Field(
        doc="The column name for the values to be plotted on the x axis.",
        dtype=str,
        default="Functor",
    )

    yColName = pexConfig.Field(
        doc="The column name for the values to be plotted on the y axis.",
        dtype=str,
        default="Functor",
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

    sourceTypeColName = pexConfig.Field(
        doc="The column to use for star - galaxy separation.",
        dtype=str,
        default="base_ClassificationExtendedness_flag",
    )

    objectsToPlot = pexConfig.Field(
        doc="Which types of objects to include on the plot, should be one of 'stars', 'galaxies' or 'all'.",
        dtype=str,
        default="stars",
    )

    zColName = pexConfig.Field(
        doc="Which column to use to color code the points.",
        dtype=str,
        default="Functor"
    )

    axisFunctors = ConfigStructField(
        doc="The functor to use to calculate the values used on each axis. Used if <axis>ColName is "
            "set to 'Functor'.",
        default={"xFunctor": RadToDeg, "yFunctor": RadToDeg, "zFunctor": SNCalculator},
    )

    selectorRegistry = ConfigStructField(
        doc="Which selectors to use to narrow down the data for QA plotting.",
        default={"mainFlagSelector": MainFlagSelector},
    )

    statisticSelectors = ConfigStructField(
        doc="Selectors to use to decide which points to use for calculating statistics.",
        default={"statSelector": LowSnSelector},
    )

    def setDefaults(self):
        self.axisFunctors.xFunctor.colName = "coord_ra"
        self.axisFunctors.yFunctor.colName = "coord_dec"


class SkyPlotTask(pipeBase.PipelineTask):

    ConfigClass = SkyPlotTaskConfig
    _DefaultName = "skyPlotTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class
        inputs = butlerQC.get(inputRefs)
        runName = inputRefs.catPlot.run
        dataId = butlerQC.quantum.dataId
        inputs["dataId"] = dataId
        inputs["runName"] = runName
        localConnections = self.config.ConnectionsClass(config=self.config)
        inputs["tableName"] = localConnections.catPlot.name
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, catPlot, dataId, runName, skymap, tableName):
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
        `self.config.selectorRegistry`.
        If the column names are 'Functor' then the functors specified in
        `self.config.axisFunctors` are used to calculate the required values.
        After this the following functions are run:

        `parsePlotInfo` which uses the dataId, runName and tableName to add
        useful information to the plot.

        `generateSummaryStats` which parses the skymap to give the corners of
        the patches for later plotting and calculates some basic statistics
        in each patch for the column in self.zColName.

        `SkyPlot` which makes the plot of the sky distribution of
        `self.zColName`.

        """
        xLabel = self.config.xLabel
        yLabel = self.config.yLabel

        # Apply the selectors to narrow down the sources to use
        mask = np.ones(len(catPlot), dtype=bool)
        for name, selector in self.config.selectorRegistry:
            mask *= selector.select(catPlot)
        catPlot = catPlot[mask]

        # Calculate extra columns as needed
        if self.config.xColName == "Functor":
            xVals, colName = self.config.axisFunctors.xFunctor.calculate(catPlot)
            catPlot.loc[:, colName] = xVals
            self.xColName = colName
        else:
            self.xColName = self.config.xColName

        if self.config.yColName == "Functor":
            yVals, colName = self.config.axisFunctors.yFunctor.calculate(catPlot)
            catPlot.loc[:, colName] = yVals
            self.yColName = colName
        else:
            self.yColName = self.config.yColName

        if self.config.zColName == "Functor":
            zVals, colName = self.config.axisFunctors.zFunctor.calculate(catPlot)
            catPlot.loc[:, colName] = zVals
            self.zColName = colName
        else:
            self.zColName = self.config.zColName

        # Decide which points to use for stats calculation
        useForStats = np.zeros(len(catPlot))
        statPoints = self.config.statisticSelectors.statSelector.select(catPlot)
        useForStats[statPoints] = 1
        catPlot.loc[:, "useForStats"] = useForStats

        # Get useful information about the plot
        plotInfo = parsePlotInfo(dataId, runName, tableName)
        # Calculate the corners of the patches and some associated stats
        sumStats = generateSummaryStats(catPlot, self.zColName, skymap, plotInfo)
        # Make the plot
        fig = self.skyPlot(catPlot, xLabel, yLabel, plotInfo, sumStats)

        return pipeBase.Struct(skyPlot=fig)

    def skyPlot(self, catPlot, xLabel, yLabel, plotInfo, sumStats):

        """Makes a generic plot showing the value at given points on the sky.

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
        sumStats : `dict`
            A dictionary where the patchIds are the keys which store the R.A.
            and dec of the corners of the patch.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The resulting figure.

        Notes
        -----
        Uses the config options `self.config.xColName` and
        `self.config.yColName` to plot points color coded by `self.zColName`.
        The points plotted are those selected by the selectors specified in
        `self.config.selectorRegistry`.
        """

        self.log.info("Plotting {}: the values of {} for {} on a sky plot.".format(
                      self.config.connections.plotName, self.zColName,
                      self.config.objectsToPlot))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Need to separate stars and galaxies
        stars = (catPlot[self.config.sourceTypeColName] == 0.0)
        galaxies = (catPlot[self.config.sourceTypeColName] == 1.0)

        # For galaxies
        xsGalaxies = catPlot[self.xColName].values[galaxies]
        ysGalaxies = catPlot[self.yColName].values[galaxies]
        colorValsGalaxies = catPlot[self.zColName].values[galaxies]

        # For stars
        xsStars = catPlot[self.xColName].values[stars]
        ysStars = catPlot[self.yColName].values[stars]
        colorValsStars = catPlot[self.zColName].values[stars]

        # Calculate some statistics
        if self.config.objectsToPlot == "galaxies" or self.config.objectsToPlot == "all":
            statGals = ((catPlot["useForStats"] == 1) & galaxies)
            statGalMed = np.nanmedian(catPlot[self.zColName].values[statGals])
            statGalMad = sigmaMad(catPlot[self.zColName].values[statGals], nan_policy="omit")

            galStatsText = ("Median: {:0.2f}\n".format(statGalMed) + r"$\sigma_{MAD}$: "
                            + "{:0.2f}".format(statGalMad))
            # Add statistics
            bbox = dict(facecolor="C1", alpha=0.3, edgecolor="none")
            ax.text(0.7, 0.92, galStatsText, transform=fig.transFigure, fontsize=8, bbox=bbox)

        if self.config.objectsToPlot == "stars" or self.config.objectsToPlot == "all":

            statStars = ((catPlot["useForStats"] == 1) & stars)
            statStarMed = np.nanmedian(catPlot[self.zColName].values[statStars])
            statStarMad = sigmaMad(catPlot[self.zColName].values[statStars], nan_policy="omit")

            starStatsText = ("Median: {:0.2f}\n".format(statStarMed) + r"$\sigma_{MAD}$: "
                             + "{:0.2f}".format(statStarMad))
            # Add statistics
            bbox = dict(facecolor="C0", alpha=0.3, edgecolor="none")
            ax.text(0.8, 0.92, starStatsText, transform=fig.transFigure, fontsize=8, bbox=bbox)

        if self.config.objectsToPlot == "stars":
            toPlotList = [(xsStars, ysStars, colorValsStars, "winter_r", "Stars")]
        elif self.config.objectsToPlot == "galaxies":
            toPlotList = [(xsGalaxies, ysGalaxies, colorValsGalaxies, "autumn_r", "Galaxies")]
        elif self.config.objectsToPlot == "all":
            toPlotList = [(xsGalaxies, ysGalaxies, colorValsGalaxies, "autumn_r", "Galaxies"),
                          (xsStars, ysStars, colorValsStars, "winter_r", "Stars")]

        # Corner plot of patches showing summary stat in each
        patches = []
        for dataId in sumStats.keys():
            (corners, stat) = sumStats[dataId]
            ra = corners[0][0].asDegrees()
            dec = corners[0][1].asDegrees()
            xy = (ra, dec)
            width = corners[2][0].asDegrees() - ra
            height = corners[2][1].asDegrees() - dec
            patches.append(Rectangle(xy, width, height, alpha=0.3))
            ras = [ra.asDegrees() for (ra, dec) in corners]
            decs = [dec.asDegrees() for (ra, dec) in corners]
            ax.plot(ras + [ras[0]], decs + [decs[0]], "k", lw=0.5)
            cenX = ra + width / 2
            cenY = dec + height / 2
            if dataId != "tract":
                ax.annotate(dataId, (cenX, cenY), color="k", fontsize=7, ha="center", va="center",
                            path_effects=[patheffects.withStroke(linewidth=2, foreground="w")])

        for (i, (xs, ys, colorVals, cmap, label)) in enumerate(toPlotList):
            med = np.median(colorVals)
            mad = sigmaMad(colorVals)
            vmin = med - 3*mad
            vmax = med + 3*mad
            scatterOut = ax.scatter(xs, ys, c=colorVals, cmap=cmap, s=10.0, vmin=vmin, vmax=vmax)
            cax = fig.add_axes([0.87 + i*0.04, 0.11, 0.04, 0.77])
            plt.colorbar(scatterOut, cax=cax, extend="both")
            colorBarLabel = "{}: {}".format(self.zColName, label)
            text = cax.text(0.6, 0.5, colorBarLabel, color="k", rotation="vertical", transform=cax.transAxes,
                            ha="center", va="center", fontsize=10)
            text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])

            if i == 0 and len(toPlotList) > 1:
                cax.yaxis.set_ticks_position("left")

        ax.set_xlabel(self.config.xLabel)
        ax.set_ylabel(self.config.yLabel)

        plt.draw()

        # Add useful information to the plot
        plt.subplots_adjust(wspace=0.0, hspace=0.0, right=0.85)
        fig = plt.gcf()
        fig = addPlotInfo(fig, plotInfo)

        return fig
