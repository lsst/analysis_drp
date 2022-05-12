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
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
from typing import List, Tuple

from lsst.geom import Box2D, SpherePoint, degrees

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


def generateSummaryStats(cat, colName, skymap, plotInfo):
    """Generate a summary statistic in each patch or detector

    Parameters
    ----------
    cat : `pandas.core.frame.DataFrame`
    colName : `str`
    skymap : `lsst.skymap.ringsSkyMap.RingsSkyMap`
    plotInfo : `dict`

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

        corners = Box2D(patchInfo.getInnerBBox()).getCorners()
        skyCoords = tractWcs.pixelToSky(corners)

        patchInfoDict[gen3PatchId] = (skyCoords, stat)

    tractCorners = Box2D(tractInfo.getBBox()).getCorners()
    skyCoords = tractWcs.pixelToSky(tractCorners)
    patchInfoDict["tract"] = (skyCoords, np.nan)

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
