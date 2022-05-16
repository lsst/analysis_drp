import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_absolute_deviation as sigmaMad
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pathEffects
from lsst.pex.config import Field
import lsst.pipe.base as pipeBase

from .plotUtils import addPlotInfo, mkColormap, extremaSort
from .skyPlot import SkyPlotTask, SkyPlotTaskConnections

__all__ = ["QuiverPlotTaskConnections", "QuiverPlotTaskConfig", "QuiverPlotTask"]


class QuiverPlotTaskConnections(SkyPlotTaskConnections):
    skyPlot = pipeBase.connectionTypes.Output(doc=("A quiver plot showing the on-sky distribution of a "
                                                   "complex value."),
                                              storageClass="Plot",
                                              name="quiverPlot_{plotName}",
                                              dimensions=("tract", "skymap"))


class QuiverPlotTaskConfig(SkyPlotTask.ConfigClass, pipelineConnections=QuiverPlotTaskConnections):

    includeQuiverKey = Field(doc=("Include a key in addition to the colorbar "
                                  "to show the scale of the quivers?"),
                             dtype=bool,
                             default=False)

    def setDefaults(self):
        super().setDefaults()


class QuiverPlotTask(SkyPlotTask):

    ConfigClass = QuiverPlotTaskConfig
    _DefaultName = "quiverPlotTask"

    def skyPlot(self, catPlot, plotInfo, sumStats):
        """Make a quiver plot of a complex quantity at given points on the sky.

        Parameters
        ----------
        catPlot : `pandas.core.frame.DataFrame`
            The catalog to plot the points from.
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
        `self.config.yColName` to plot points color coded by
        `self.config.axisActions['zAction']`.
        The points plotted are those selected by the selectors specified in
        `self.config.selectorActions`.
        """
        # TODO: DM-34057 - the notes in the above docstring aren't quite right.
        self.log.info("Plotting {}: the values of {} on a quiver plot.".format(
                      self.config.connections.plotName, self.config.axisLabels["z"]))

        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111, aspect=1.0)
        # Quiver plots must have an aspect ratio of 1:1

        # Make divergent colormaps for stars, galaxes and all the points
        blueGreen = mkColormap(["midnightblue", "lightcyan", "darkgreen"])
        redPurple = mkColormap(["indigo", "lemonchiffon", "firebrick"])
        orangeBlue = mkColormap(["darkOrange", "thistle", "midnightblue"])

        # Need to separate stars and galaxies
        stars = (catPlot["sourceType"] == 1)
        galaxies = (catPlot["sourceType"] == 2)

        xCol = self.config.axisLabels["x"]
        yCol = self.config.axisLabels["y"]
        zCol = self.config.axisLabels["z"]

        # TODO: Use better names instead of colorValsGalaxies in DM-34057.
        # For galaxies
        colorValsGalaxies = catPlot.loc[galaxies, zCol].values
        ids = extremaSort(colorValsGalaxies)
        xsGalaxies = catPlot.loc[galaxies, xCol].values[ids]
        ysGalaxies = catPlot.loc[galaxies, yCol].values[ids]
        colorValsGalaxies = colorValsGalaxies[ids]

        # For stars
        colorValsStars = catPlot.loc[stars, zCol].values
        ids = extremaSort(colorValsStars)
        xsStars = catPlot.loc[stars, xCol].values[ids]
        ysStars = catPlot.loc[stars, yCol].values[ids]
        colorValsStars = colorValsStars[ids]

        # Calculate some statistics
        if np.any(catPlot["sourceType"] == 2):
            statGals = ((catPlot["useForStats"] == 1) & galaxies)
            statGalMed = np.nanmedian(catPlot.loc[statGals, zCol])
            statGalMad = sigmaMad(catPlot.loc[statGals, zCol], nan_policy="omit")

            galStatsText = ("Median: {:.2f}\n".format(statGalMed) + r"$\sigma_{MAD}$: "
                            + "{:.2f}\n".format(statGalMad) + r"n$_{points}$: "
                            + "{}".format(len(xsGalaxies)))
            # Add statistics
            bbox = dict(facecolor="lemonchiffon", alpha=0.5, edgecolor="none")
            # Check if plotting stars and galaxies, if so move the
            # text box so that both can be seen. Needs to be
            # > 2 becuase not being plotted points are assigned 0
            if len(list(set(catPlot["sourceType"].values))) > 2:
                boxLoc = 0.63
            else:
                boxLoc = 0.8
            ax.text(boxLoc, 0.91, galStatsText, transform=fig.transFigure, fontsize=8, bbox=bbox)

        if np.any(catPlot["sourceType"] == 1):

            statStars = ((catPlot["useForStats"] == 1) & stars)
            statStarMed = np.nanmedian(catPlot.loc[statStars, zCol])
            statStarMad = sigmaMad(catPlot.loc[statStars, zCol], nan_policy="omit")

            starStatsText = ("Median: {:.2f}\n".format(statStarMed) + r"$\sigma_{MAD}$: "
                             + "{:.2f}\n".format(statStarMad) + r"n$_{points}$: "
                             + "{}".format(len(xsStars)))
            # Add statistics
            bbox = dict(facecolor="paleturquoise", alpha=0.5, edgecolor="none")
            ax.text(0.8, 0.91, starStatsText, transform=fig.transFigure, fontsize=8, bbox=bbox)

        if np.any(catPlot["sourceType"] == 10):

            statAll = (catPlot["useForStats"] == 1)
            statAllMed = np.nanmedian(catPlot.loc[statAll, zCol])
            statAllMad = sigmaMad(catPlot.loc[statAll, zCol], nan_policy="omit")

            allStatsText = ("Median: {:.2f}\n".format(statAllMed) + r"$\sigma_{MAD}$: "
                            + "{:.2f}\n".format(statAllMad) + r"n$_{points}$: "
                            + "{}".format(len(catPlot)))
            bbox = dict(facecolor="purple", alpha=0.2, edgecolor="none")
            ax.text(0.8, 0.91, allStatsText, transform=fig.transFigure, fontsize=8, bbox=bbox)

        if np.any(catPlot["sourceType"] == 9):

            statAll = (catPlot["useForStats"] == 1)
            statAllMed = np.nanmedian(catPlot.loc[statAll, zCol])
            statAllMad = sigmaMad(catPlot.loc[statAll, zCol], nan_policy="omit")

            allStatsText = ("Median: {:.2f}\n".format(statAllMed) + r"$\sigma_{MAD}$: "
                            + "{:.2f}\n".format(statAllMad) + r"n$_{points}$: "
                            + "{}".format(len(catPlot)))
            bbox = dict(facecolor="green", alpha=0.2, edgecolor="none")
            ax.text(0.8, 0.91, allStatsText, transform=fig.transFigure, fontsize=8, bbox=bbox)

        toPlotList = []
        if np.any(catPlot["sourceType"] == 1):
            toPlotList.append((xsStars, ysStars, colorValsStars, blueGreen, "Stars"))
        if np.any(catPlot["sourceType"] == 2):
            toPlotList.append((xsGalaxies, ysGalaxies, colorValsGalaxies, redPurple, "Galaxies"))
        if np.any(catPlot["sourceType"] == 10):
            ids = extremaSort(catPlot[zCol].values)
            toPlotList.append((catPlot[xCol].values[ids], catPlot[yCol].values[ids],
                               catPlot[zCol].values[ids], orangeBlue, "All"))
        if np.any(catPlot["sourceType"] == 9):
            toPlotList.append((catPlot[xCol], catPlot[yCol], catPlot[zCol], "viridis", "Unknown"))

        # Corner plot of patches showing summary stat in each
        if self.config.plotOutlines:
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
                if dataId == "tract":
                    minRa = np.min(ras)
                    minDec = np.min(decs)
                    maxRa = np.max(ras)
                    maxDec = np.max(decs)
                if dataId != "tract":
                    ax.annotate(dataId, (cenX, cenY), color="k", fontsize=7, ha="center", va="center",
                                path_effects=[pathEffects.withStroke(linewidth=2, foreground="w")])

        for (i, (xs, ys, eVals, cmap, label)) in enumerate(toPlotList):
            if "tract" not in sumStats.keys() or not self.config.plotOutlines:
                minRa = np.min(xs)
                maxRa = np.max(xs)
                minDec = np.min(ys)
                maxDec = np.max(ys)
            e, e1, e2 = np.abs(eVals), np.real(eVals), np.imag(eVals)
            quiverOut = ax.quiver(xs, ys, e1, e2, e, angles="uv", scale=None, pivot="middle",
                                  units="width", width=0.001, headwidth=0.0, headlength=0.0,
                                  headaxislength=0.0, label=label, cmap=cmap)

            if self.config.includeQuiverKey:
                # Should the quiverKey go into plotInfo?
                # TODO: DM-35047 must generalize this to non-ellipticity plots.
                ax.quiverkey(quiverOut, 0.1, 0.1, 10*abs(statStarMad), angle=0,
                             label=r"$10\times\sigma_{MAD}$" + " = {:.3f}".format(10*abs(statStarMad)),
                             labelpos="E", coordinates="figure", color="r", labelcolor="g", lw=2)
                ax.quiverkey(quiverOut, 0.1, 0.1, 10*abs(statStarMad), angle=45,
                             label=r"$10\times\sigma_{MAD}$" + " = {:.3f}".format(10*abs(statStarMad)),
                             labelpos="E", coordinates="figure", color="r", labelcolor="g", lw=2)

            cax = fig.add_axes([0.87 + i*0.04, 0.11, 0.04, 0.77])
            plt.colorbar(quiverOut, cax=cax, extend="both")
            colorBarLabel = "{}: {}".format(self.config.axisLabels["z"], label)
            text = cax.text(0.5, 0.5, colorBarLabel, color="k", rotation="vertical", transform=cax.transAxes,
                            ha="center", va="center", fontsize=10)
            text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])
            cax.tick_params(labelsize=7)

            if i == 0 and len(toPlotList) > 1:
                cax.yaxis.set_ticks_position("left")

        ax.set_xlabel(xCol)
        ax.set_ylabel(yCol)
        ax.tick_params(axis="x", labelrotation=25)
        ax.tick_params(labelsize=7)

        ax.set_aspect("equal")
        plt.draw()

        # Find some useful axis limits
        lenXs = [len(xs) for (xs, _, _, _, _) in toPlotList]
        if lenXs != [] and np.max(lenXs) > 1000:
            padRa = (maxRa - minRa)/10
            padDec = (maxDec - minDec)/10
            ax.set_xlim(maxRa + padRa, minRa - padRa)
            ax.set_ylim(minDec - padDec, maxDec + padDec)
        else:
            ax.invert_xaxis()

        # Add useful information to the plot
        plt.subplots_adjust(wspace=0.0, hspace=0.0, right=0.85)
        fig = plt.gcf()
        fig = addPlotInfo(fig, plotInfo)

        return fig
