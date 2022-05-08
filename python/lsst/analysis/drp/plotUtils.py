# This file is part of analysis_drp.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import matplotlib
import matplotlib.patheffects as pathEffects
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.odr as scipyODR
from typing import List, Tuple

from lsst.geom import Box2D, SpherePoint, degrees
import lsst.pipe.base as pipeBase

null_formatter = matplotlib.ticker.NullFormatter()


def parsePlotInfo(dataId, runName, tableName, bands, plotName, SN):
    """Parse plot info from the dataId

    Parameters
    ----------
    dataId : `lsst.daf.butler.core.dimensions.`
             `_coordinate._ExpandedTupleDataCoordinate`
    runName : `str`

    Returns
    -------
    plotInfo : `dict`
    """
    plotInfo = {"run": runName, "tractTableType": tableName, "plotName": plotName, "SN": SN}

    for dataInfo in dataId:
        plotInfo[dataInfo.name] = dataId[dataInfo.name]

    bandStr = ""
    for band in bands:
        bandStr += (", " + band)
    plotInfo["bands"] = bandStr[2:]

    if "tract" not in plotInfo.keys():
        plotInfo["tract"] = "N/A"
    if "visit" not in plotInfo.keys():
        plotInfo["visit"] = "N/A"

    return plotInfo


def generateSummaryStats(cat, colName, skymap, plotInfo, patchesWithData):
    """Generate a summary statistic in each patch or detector

    Parameters
    ----------
    cat : `pandas.core.frame.DataFrame`
    colName : `str`
    skymap : `lsst.skymap.ringsSkyMap.RingsSkyMap`
    plotInfo : `dict`
    patchesWithData : `numpy.ndarray`

    Returns
    -------
    patchInfoDict : `dict`
    """

    # TODO: what is the more generic type of skymap?
    tractInfo = skymap.generateTract(plotInfo["tract"])
    tractWcs = tractInfo.getWcs()

    if "sourceType" in cat.columns:
        cat = cat.loc[cat["sourceType"] != 0]

    # For now also convert the gen 2 patchIds to gen 3

    patchInfoDict = {}
    maxPatchNum = tractInfo.num_patches.x*tractInfo.num_patches.y
    patches = np.arange(0, maxPatchNum, 1)
    dataLimitsBbox = None
    for patch in patches:
        if patch is None:
            continue
        # Once the objectTable_tract catalogues are using gen 3 patches
        # this will go away
        onPatch = (cat["patch"] == patch)
        stat = np.nanmedian(cat[colName].values[onPatch])
        try:
            patchTuple = (int(patch.split(",")[0]), int(patch.split(",")[-1]))
            patchInfo = tractInfo.getPatchInfo(patchTuple)
            gen3PatchId = tractInfo.getSequentialPatchIndex(patchInfo)
        except AttributeError:
            # For native gen 3 tables the patches don't need converting
            # When we are no longer looking at the gen 2 -> gen 3
            # converted repos we can tidy this up
            gen3PatchId = patch
            patchInfo = tractInfo.getPatchInfo(patch)

        innerBbox = Box2D(patchInfo.getInnerBBox())
        corners = innerBbox.getCorners()
        skyCoords = tractWcs.pixelToSky(corners)

        patchInfoDict[gen3PatchId] = (skyCoords, stat)
        # Make a BBox that includes all patches with data
        if patch in patchesWithData:
            if dataLimitsBbox is None:
                dataLimitsBbox = innerBbox
            else:
                dataLimitsBbox.include(innerBbox)

    tractCorners = Box2D(tractInfo.getBBox()).getCorners()
    skyCoords = tractWcs.pixelToSky(tractCorners)
    patchInfoDict["tract"] = (skyCoords, np.nan)
    if dataLimitsBbox is not None:
        dataCoords = tractWcs.pixelToSky(dataLimitsBbox.getCorners())
    else:
        # If no data was found, fall back to the full tract boundary
        dataCoords = skyCoords
    patchInfoDict["dataLimits"] = (dataCoords, np.nan)
    patchInfoDict["patchesWithData"] = patchesWithData

    return patchInfoDict


def generateSummaryStatsVisit(cat, colName, visitSummaryTable, plotInfo):
    """Generate a summary statistic in each patch or detector

    Parameters
    ----------
    cat : `pandas.core.frame.DataFrame`
    colName : `str`
    visitSummaryTable : `pandas.core.frame.DataFrame`
    plotInfo : `dict`

    Returns
    -------
    visitInfoDict : `dict`
    """

    visitInfoDict = {}
    for ccd in cat.detector.unique():
        if ccd is None:
            continue
        onCcd = (cat["detector"] == ccd)
        stat = np.nanmedian(cat[colName].values[onCcd])

        sumRow = (visitSummaryTable["id"] == ccd)
        corners = zip(visitSummaryTable["raCorners"][sumRow][0], visitSummaryTable["decCorners"][sumRow][0])
        cornersOut = []
        for (ra, dec) in corners:
            corner = SpherePoint(ra, dec, units=degrees)
            cornersOut.append(corner)

        visitInfoDict[ccd] = (cornersOut, stat)

    return visitInfoDict


# Inspired by matplotlib.testing.remove_ticks_and_titles
def get_and_remove_axis_text(ax) -> Tuple[List[str], List[np.ndarray]]:
    """Remove text from an Axis and its children and return with line points.

    Parameters
    ----------
    ax : `plt.Axis`
        A matplotlib figure axis.

    Returns
    -------
    texts : `List[str]`
        A list of all text strings (title and axis/legend/tick labels).
    line_xys : `List[numpy.ndarray]`
        A list of all line ``_xy`` attributes (arrays of shape ``(N, 2)``).
    """
    line_xys = [line._xy for line in ax.lines]
    texts = [text.get_text() for text in (ax.title, ax.xaxis.label, ax.yaxis.label)]
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    try:
        texts_legend = ax.get_legend().texts
        texts.extend(text.get_text() for text in texts_legend)
        for text in texts_legend:
            text.set_alpha(0)
    except AttributeError:
        pass

    for idx in range(len(ax.texts)):
        texts.append(ax.texts[idx].get_text())
        ax.texts[idx].set_text('')

    ax.xaxis.set_major_formatter(null_formatter)
    ax.xaxis.set_minor_formatter(null_formatter)
    ax.yaxis.set_major_formatter(null_formatter)
    ax.yaxis.set_minor_formatter(null_formatter)
    try:
        ax.zaxis.set_major_formatter(null_formatter)
        ax.zaxis.set_minor_formatter(null_formatter)
    except AttributeError:
        pass
    for child in ax.child_axes:
        texts_child, lines_child = get_and_remove_axis_text(child)
        texts.extend(texts_child)

    return texts, line_xys


def get_and_remove_figure_text(figure: plt.Figure):
    """Remove text from a Figure and its Axes and return with line points.

    Parameters
    ----------
    figure : `matplotlib.pyplot.Figure`
        A matplotlib figure.

    Returns
    -------
    texts : `List[str]`
        A list of all text strings (title and axis/legend/tick labels).
    line_xys : `List[numpy.ndarray]`, (N, 2)
        A list of all line ``_xy`` attributes (arrays of shape ``(N, 2)``).
    """
    texts = [str(figure._suptitle)]
    lines = []
    figure.suptitle("")

    texts.extend(text.get_text() for text in figure.texts)
    figure.texts = []

    for ax in figure.get_axes():
        texts_ax, lines_ax = get_and_remove_axis_text(ax)
        texts.extend(texts_ax)
        lines.extend(lines_ax)

    return texts, lines


def addPlotInfo(fig, plotInfo):
    """Add useful information to the plot

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
    plotInfo : `dict`

    Returns
    -------
    fig : `matplotlib.figure.Figure`
    """

    # TO DO: figure out how to get this information
    photocalibDataset = "None"
    astroDataset = "None"

    plt.text(0.01, 0.99, plotInfo["plotName"], fontsize=8, transform=fig.transFigure, ha="left", va="top")

    run = plotInfo["run"]
    datasetsUsed = f"\nPhotoCalib: {photocalibDataset}, Astrometry: {astroDataset}"
    tableType = f"\nTable: {plotInfo['tractTableType']}"

    dataIdText = ""
    if str(plotInfo["tract"]) != "N/A":
        dataIdText += f", Tract: {plotInfo['tract']}"
    if str(plotInfo["visit"]) != "N/A":
        dataIdText += f", Visit: {plotInfo['visit']}"

    bandsText = f", Bands: {''.join(plotInfo['bands'].split(' '))}"
    SNText = f", S/N: {plotInfo['SN']}"
    infoText = f"\n{run}{datasetsUsed}{tableType}{dataIdText}{bandsText}{SNText}"
    plt.text(0.01, 0.98, infoText, fontsize=7, transform=fig.transFigure, alpha=0.6, ha="left", va="top")

    return fig


def stellarLocusFit(xs, ys, paramDict):
    """Make a fit to the stellar locus

    Parameters
    ----------
    xs : `numpy.ndarray`
        The color on the xaxis
    ys : `numpy.ndarray`
        The color on the yaxis
    paramDict : lsst.pex.config.dictField.Dict
        A dictionary of parameters for line fitting
        xMin : `float`
            The minimum x edge of the box to use for initial fitting
        xMax : `float`
            The maximum x edge of the box to use for initial fitting
        yMin : `float`
            The minimum y edge of the box to use for initial fitting
        yMax : `float`
            The maximum y edge of the box to use for initial fitting
        mHW : `float`
            The hardwired gradient for the fit
        bHW : `float`
            The hardwired intercept of the fit

    Returns
    -------
    paramsOut : `dict`
        A dictionary of the calculated fit parameters
        xMin : `float`
            The minimum x edge of the box to use for initial fitting
        xMax : `float`
            The maximum x edge of the box to use for initial fitting
        yMin : `float`
            The minimum y edge of the box to use for initial fitting
        yMax : `float`
            The maximum y edge of the box to use for initial fitting
        mHW : `float`
            The hardwired gradient for the fit
        bHW : `float`
            The hardwired intercept of the fit
        mODR : `float`
            The gradient calculated by the ODR fit
        bODR : `float`
            The intercept calculated by the ODR fit
        yBoxMin : `float`
            The y value of the fitted line at xMin
        yBoxMax : `float`
            The y value of the fitted line at xMax
        bPerpMin : `float`
            The intercept of the perpendicular line that goes through xMin
        bPerpMax : `float`
            The intercept of the perpendicular line that goes through xMax
        mODR2 : `float`
            The gradient from the second round of fitting
        bODR2 : `float`
            The intercept from the second round of fitting
        mPerp : `float`
            The gradient of the line perpendicular to the line from the
            second fit

    Notes
    -----
    The code does two rounds of fitting, the first is initiated using the
    hardwired values given in the `paramDict` parameter and is done using
    an Orthogonal Distance Regression fit to the points defined by the
    box of xMin, xMax, yMin and yMax. Once this fitting has been done a
    perpendicular bisector is calculated at either end of the line and
    only points that fall within these lines are used to recalculate the fit.
    """

    # Points to use for the fit
    fitPoints = np.where((xs > paramDict["xMin"]) & (xs < paramDict["xMax"])
                         & (ys > paramDict["yMin"]) & (ys < paramDict["yMax"]))[0]

    linear = scipyODR.polynomial(1)

    data = scipyODR.Data(xs[fitPoints], ys[fitPoints])
    odr = scipyODR.ODR(data, linear, beta0=[paramDict["bHW"], paramDict["mHW"]])
    params = odr.run()
    mODR = float(params.beta[1])
    bODR = float(params.beta[0])

    paramsOut = {"xMin": paramDict["xMin"], "xMax": paramDict["xMax"], "yMin": paramDict["yMin"],
                 "yMax": paramDict["yMax"], "mHW": paramDict["mHW"], "bHW": paramDict["bHW"],
                 "mODR": mODR, "bODR": bODR}

    # Having found the initial fit calculate perpendicular ends
    mPerp = -1.0/mODR
    # When the gradient is really steep we need to use
    # the y limits of the box rather than the x ones

    if np.abs(mODR) > 1:
        yBoxMin = paramDict["yMin"]
        xBoxMin = (yBoxMin - bODR)/mODR
        yBoxMax = paramDict["yMax"]
        xBoxMax = (yBoxMax - bODR)/mODR
    else:
        yBoxMin = mODR*paramDict["xMin"] + bODR
        xBoxMin = paramDict["xMin"]
        yBoxMax = mODR*paramDict["xMax"] + bODR
        xBoxMax = paramDict["xMax"]

    bPerpMin = yBoxMin - mPerp*xBoxMin

    paramsOut["yBoxMin"] = yBoxMin
    paramsOut["bPerpMin"] = bPerpMin

    bPerpMax = yBoxMax - mPerp*xBoxMax

    paramsOut["yBoxMax"] = yBoxMax
    paramsOut["bPerpMax"] = bPerpMax

    # Use these perpendicular lines to chose the data and refit
    fitPoints = ((ys > mPerp*xs + bPerpMin) & (ys < mPerp*xs + bPerpMax))
    data = scipyODR.Data(xs[fitPoints], ys[fitPoints])
    odr = scipyODR.ODR(data, linear, beta0=[bODR, mODR])
    params = odr.run()
    mODR = float(params.beta[1])
    bODR = float(params.beta[0])

    paramsOut["mODR2"] = float(params.beta[1])
    paramsOut["bODR2"] = float(params.beta[0])

    paramsOut["mPerp"] = -1.0/paramsOut["mODR2"]

    return paramsOut


def perpDistance(p1, p2, points):
    """Calculate the perpendicular distance to a line from a point

    Parameters
    ----------
    p1 : `numpy.ndarray`
        A point on the line
    p2 : `numpy.ndarray`
        Another point on the line
    points : `zip`
        The points to calculate the distance to

    Returns
    -------
    dists : `list`
        The distances from the line to the points. Uses the cross
        product to work this out.
    """
    dists = []
    for point in points:
        point = np.array(point)
        distToLine = np.cross(p2 - p1, point - p1)/np.linalg.norm(p2 - p1)
        dists.append(distToLine)

    return dists


def mkColormap(colorNames):
    """Make a colormap from the list of color names.

    Parameters
    ----------
    colorNames : `list`
        A list of strings that correspond to matplotlib
        named colors.

    Returns
    -------
    cmap : `matplotlib.colors.LinearSegmentedColormap`
    """

    nums = np.linspace(0, 1, len(colorNames))
    blues = []
    greens = []
    reds = []
    for (num, color) in zip(nums, colorNames):
        r, g, b = colors.colorConverter.to_rgb(color)
        blues.append((num, b, b))
        greens.append((num, g, g))
        reds.append((num, r, r))

    colorDict = {"blue": blues, "red": reds, "green": greens}
    cmap = colors.LinearSegmentedColormap("newCmap", colorDict)
    return cmap


def extremaSort(xs):
    """Return the ids of the points reordered so that those
    furthest from the median, in absolute terms, are last.

    Parameters
    ----------
    xs : `np.array`
        An array of the values to sort

    Returns
    -------
    ids : `np.array`
    """

    med = np.median(xs)
    dists = np.abs(xs - med)
    ids = np.argsort(dists)
    return ids


def plotPatchOutlines(ax, sumStats, limitsKey="dataLimits", forcePatchLabel=True, colorByPatchStat=False,
                      fig=None, histPlot=False):
    """Plot the patch outlines on the given axis.

    Parameters
    ----------
    ax : `plt.Axis`
        A matplotlib figure axis.
    sumStats : `dict`
        A dictionary where the patchIds are the keys which store the R.A.
        and dec of the corners of the patch.
    limitsKey : `str`, optional
        Trims the returned plot axis limits to the area covering patches
        that have data if equal to "dataLimits".  Sets returned plot limits
        to the tract boundary if equal to "tract".  Currently can only be
        one of "dataLimits" or "tract".

    Returns
    -------
    plotLimits : `lsst.pipe.base.Struct`
        Results struct with attribute:

            ``minRa``
                The minimum RA extent for the plot (in degrees).  Trimmed to
                the area with data present if ``limitsKey`` is "dataLimits".
            ``maxRa``
                The maximum RA extent for the plot (in degrees).  Trimmed to
                the area with data present if ``limitsKey`` is "dataLimits".
            ``minDec``
                The minimum Dec extent for the plot (in degrees).  Trimmed to
                the area with data present if ``limitsKey`` is "dataLimits".
            ``maxDec``
                The maximum Dec extent for the plot (in degrees).  Trimmed to
                the area with data present if ``limitsKey`` is "dataLimits".

    Raises
    ------
    ValueError
        If the ``limitsKey`` is set to a value that is not one of the
        ``sumStats`` keys.
    """
    if limitsKey not in sumStats.keys():
        raise ValueError(f"Value of limitKey must be one included in sumStats.keys().  "
                         f"Was given: {limitsKey}.")

    patchesWithData = sumStats.pop("patchesWithData")
    patches = []
    colors = []
    minFontSize = 4 if colorByPatchStat else 5
    for dataId in sumStats.keys():
        (corners, stat) = sumStats[dataId]
        ra = corners[0][0].asDegrees()
        dec = corners[0][1].asDegrees()
        ras = [ra.asDegrees() for (ra, dec) in corners]
        decs = [dec.asDegrees() for (ra, dec) in corners]
        if dataId == limitsKey:
            minRa = np.min(ras)
            minDec = np.min(decs)
            maxRa = np.max(ras)
            maxDec = np.max(decs)
        if dataId != "dataLimits":
            if not (limitsKey == "dataLimits" and dataId == "tract"):
                ax.plot(ras + [ras[0]], decs + [decs[0]], "k", lw=0.5)
            if dataId != "tract":
                if limitsKey == "tract" or (limitsKey != "tract" and dataId in patchesWithData):
                    xy = (ra, dec)
                    width = corners[2][0].asDegrees() - ra
                    height = corners[2][1].asDegrees() - dec
                    patches.append(Rectangle(xy, width, height, alpha=0.3))
                    colors.append(stat)
                    cenX = ra + width / 2
                    cenY = dec + height / 2
                    if forcePatchLabel or len(patchesWithData) <= 81:
                        if colorByPatchStat:
                            pathEffectsObj = None
                        else:
                            pathEffectsObj = [pathEffects.withStroke(linewidth=2, foreground="w")]
                        ax.annotate(dataId, (cenX, cenY), color="k", fontsize=minFontSize, ha="center",
                                    va="center", path_effects=pathEffectsObj)

    # Find and set useful axis limits
    padRa = (maxRa - minRa)/20
    padDec = (maxDec - minDec)/20
    ax.set_xlim(maxRa + padRa, minRa - padRa)
    ax.set_ylim(minDec - padDec, maxDec + padDec)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=minFontSize + 1, length=2, pad=1)
    ax.set_xlabel("R.A. (deg)", fontsize=minFontSize + 2, labelpad=2)
    ax.set_ylabel("Dec. (deg)", fontsize=minFontSize + 2, labelpad=2)

    # Add colormapping for per-patch metric coloring
    collection = None
    if colorByPatchStat:
        cmapPatch = plt.cm.coolwarm.copy()
        cmapPatch.set_bad(color="none")
        # Set the bad color to transparent and make a masked array
        colors = np.ma.array(colors, mask=np.isnan(colors))
        collection = PatchCollection(patches, cmap=cmapPatch)
        collection.set_array(colors)
        ax.add_collection(collection)

        # Add a colorbar
        if histPlot:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top", size="12%", pad="5%")
        else:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            pos = ax.get_position()
            delY = pos.y1 - pos.y0
            cax = fig.add_axes([pos.x0, pos.y0 + 1.3*delY, pos.x1 - pos.x0, 0.15*delY])
        plt.colorbar(collection, cax=cax, orientation="horizontal")
        cax.text(0.5, 0.5, "Median Value", color="k", transform=cax.transAxes, rotation="horizontal",
                 horizontalalignment="center", verticalalignment="center", fontsize=minFontSize + 1)
        cax.tick_params(axis="x", labelsize=minFontSize + 1, labeltop=True, labelbottom=False, bottom=False,
                        top=True, pad=0.5, length=2)

    plotLimits = pipeBase.Struct(
        minRa=minRa,
        maxRa=maxRa,
        minDec=minDec,
        maxDec=maxDec,
        collection=collection,
    )
    return plotLimits
