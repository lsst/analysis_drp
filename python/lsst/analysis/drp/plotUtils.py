import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as scipyODR

from lsst.geom import Box2D


def parsePlotInfo(dataId, runName, tableName):
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
    plotInfo = {"run": runName, "tractTableType": tableName}

    for dataInfo in dataId:
        plotInfo[dataInfo.name] = dataId[dataInfo.name]

    if "filter" not in plotInfo.keys():
        plotInfo["filter"] = "N/A"

    return plotInfo


def generateSummaryStats(cat, colName, skymap, plotInfo):
    """Generate a summary statistic in each patch

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
    for patch in cat.patch.unique():
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

    plt.text(0.02, 0.98, "Run:" + plotInfo["run"], fontsize=8, alpha=0.8, transform=fig.transFigure)
    datasetType = ("Datasets used: photocalib: " + photocalibDataset + ", astrometry: " + astroDataset)
    tableType = "Table: " + plotInfo["tractTableType"]
    dataIdText = "Tract: " + str(plotInfo["tract"]) + " , Filter: " + plotInfo["filter"]
    plt.text(0.02, 0.95, datasetType, fontsize=8, alpha=0.8, transform=fig.transFigure)

    plt.text(0.02, 0.92, tableType, fontsize=8, alpha=0.8, transform=fig.transFigure)
    plt.text(0.02, 0.89, dataIdText, fontsize=8, alpha=0.8, transform=fig.transFigure)
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

    if np.fabs(mODR) > 1:
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
        distToLine = np.cross(p1 - point, p2 - point)/np.linalg.norm(p2 - point)
        dists.append(distToLine)

    return dists
