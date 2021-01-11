import numpy as np
import matplotlib.pyplot as plt
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
    for patch in cat.patchId.unique():
        if patch is None:
            continue
        # Once the objectTable_tract catalogues are using gen 3 patches
        # this will go away
        onPatch = (cat["patchId"] == patch)
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
