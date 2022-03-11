import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as scipyODR
from matplotlib import colors

from lsst.geom import Box2D, SpherePoint, degrees


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
