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

from abc import abstractmethod
import pandas as pd

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

__all__ = ["MakeForcedQaTractTablesTaskConnections", "MakeForcedQaTractTablesTaskConfig",
           "MakeForcedQaTractTablesTask", "MakeUnforcedQaTractTablesTaskConnections",
           "MakeUnforcedQaTractTablesTaskConfig", "MakeUnforcedQaTractTablesTask"]


class BaseMakeQaTractTablesTaskConnections(pipeBase.PipelineTaskConnections,
                                           defaultTemplates={"coaddName": "deep"},
                                           dimensions=("tract", "skymap")):
    inputObjCats = cT.Input(
        doc="The coadd catalog from which to collect relevant columns.",
        storageClass="DataFrame",
        name="{coaddName}Coadd_obj",
        dimensions=("tract", "patch", "skymap"),
        multiple=True,
        deferLoad=True,
    )
    inputObjectTable = cT.Input(
        doc=("The objectTable_tract associated with the tract.  Only used to ensure the tables persisted "
             "have the same row ordering as the objectTable_tract tables for ease of joint use."),
        storageClass="DataFrame",
        name="objectTable_tract",
        dimensions=("tract", "skymap"),
        deferLoad=True,
    )
    qaTractTable = cT.Output(
        doc="The collated catalog of measurements with additional columns for QA added.",
        storageClass="DataFrame",
        name="qaTractTable_forced",
        dimensions=("tract", "band", "skymap"),
    )


class BaseMakeQaTractTablesTaskConfig(pipeBase.PipelineTaskConfig,
                                      pipelineConnections=BaseMakeQaTractTablesTaskConnections):
    # TODO: remove the filterMap config song and dance once DM-28475 lands
    filterMap = pexConfig.DictField(
        keytype=str, itemtype=str,
        default={},
        doc=("Dictionary mapping canonical band to the camera's physical filter.  This is currently "
             "required for {coaddName}Coadd_obj that are indexed on the latter as the dataId supplied "
             "will only contain the former (this will likely become unnecessary once the postprocess "
             "scripts are updated to index on band).")
    )
    baseColStrList = pexConfig.ListField(
        dtype=str,
        default=["coord", "tract", "patch", "visit", "ccd", "base_PixelFlags", "base_GaussianFlux",
                 "base_PsfFlux", "base_CircularApertureFlux_9_0_instFlux", "base_CircularApertureFlux_12_0",
                 "base_CircularApertureFlux_25_0", "ext_photometryKron_KronFlux", "modelfit_CModel",
                 "base_Sdss", "slot_Centroid", "slot_Shape", "ext_shapeHSM_HsmSourceMoments_",
                 "ext_shapeHSM_HsmPsfMoments_", "ext_shapeHSM_HsmShapeRegauss_", "base_Footprint",
                 "base_FPPosition", "base_ClassificationExtendedness", "parent", "detect", "deblend_nChild",
                 "base_Blendedness_abs", "base_Blendedness_flag", "base_InputCount",
                 "merge_peak_sky", "merge_measurement", "calib", "sky_source"],
        doc=("List of \"prefix\" strings of column names to load from deepCoadd_obj parquet table. "
             "All columns that start with one of these strings will be loaded UNLESS the full column "
             "name contains one of the strings listed in the notInColumnStrList config.")
    )
    notInColStrList = pexConfig.ListField(
        dtype=str,
        default=["flag_bad", "flag_no", "missingDetector_flag", "_region_", "Truncated", "_radius",
                 "_bad_", "initial", "_exp_", "_dev_", "fracDev", "objective", "SdssCentroid_flag_",
                 "SdssShape_flag_u", "SdssShape_flag_m", "_Cov", "_child_", "_parent_"],
        doc=("List of substrings to select against when creating list of columns to load from the "
             "deepCoadd_obj parquet table.")
    )


class BaseMakeQaTractTablesTask(pipeBase.PipelineTask):
    """Read in {coaddName}Coadd_obj tables and aggregate QA columns by tract.

    For a given tract, read in all of the per-patch {coaddName}Coadd_obj
    tables, select on a subset of columns that are useful for QA analyses and
    plotting routines, aggregate these by tract and persist them as datasets.
    A separate table is created for each band.  These serve the purpose of
    providing columns that are useful for QA but are not (yet?) included in the
    DPDD-based per-tract objectTable_tract tables.

    For ease of use with along with the objectTable_tract tables (e.g. being
    able to use a single boolean expression to filter on both), the columns of
    these new tables are sorted in the same order as in the objectTable_tract
    tables and, since we don't want to make assumptions on that ordering, we
    load in the table itself to do the sorting.

    Subclasses must implement their own collateTable() method to collect the
    relevant columns from the input tables.
    """

    ConfigClass = BaseMakeQaTractTablesTaskConfig
    _DefaultName = "baseMakeQaTractTablesTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs["tract"] = butlerQC.quantum.dataId["tract"]
        inputs["band"] = outputRefs.qaTractTable.dataId["band"]
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    @abstractmethod
    def makeLoadList(self, inputCatCols, activeFilterName):
        """Make a "load list" instance appropriate for this derived class.

        The list of columns to load for a given measurement flavor is derived
        at runtime based on columns existing in the input {coaddName}Coadd_obj
        and a set of glob (and anti-glob) patterns set in the configs.  This
        method is to create the appropriate list only once and must be
        implemented by all subclasses.

        Parameters
        ----------
        inputCatCols : `pandas.core.indexes.multi.MultiIndex`
            A pandas multiIndex object.  The format is assumed to be a list of
            tuples of (dataset, filter, column), from which the list of columns
            associated ``activeFilterName`` is to be isolated as the list under
            consideration.
        actiaveFilterName : `str`
            The name of the filter index key of interest.

        Returns
        -------
        qaLoadListInstance :
              `lsst.analysis.drp.makeQaTractTables.ForcedQaLoadList` or
              `lsst.analysis.drp.makeQaTractTables.UnforcedQaLoadList`
            An instance of the class called to derive the column list
            definitions, which are set as class attributes.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    @abstractmethod
    def collateTable(self, inputCat, loadListObject, activeFilterName, filterName, haveFilter):
        """Collect and aggregate catalog columns required for QA purposes.

        Parameters
        ----------
        inputCat : `lsst.daf.butler.DeferredDatasetHandle` of
                   `pandas.core.frame.DataFrame`
            The input _obj table from which to collect the relevant data
            columns.
        loadListObject : `lsst.analysis.drp.makeQaTractTables.ForcedQaLoadList`
            Class object containing the load lists required as instance
            attribute.
        actiaveFilterName : `str`
            The name of the filter index key.  The "active" prefix is used to
            indicate that this may be a stand-in filter for a given patch for
            which data for the true filter of interest does not exist, as
            indicated by the value of ``haveFilter``.  In this case, the
            stand-in is used as a template for the columns to be added, but
            they will all be populated by NaNs.
        filterName : `str`
            The name of the filter index key for the band of interest.  Will be
            the same as ``activeFilterName`` if ``haveFilter`` is `True`.
        haveFilter : `bool`
            A `bool` indicating if the current input table has an entry for
            the filter of interest.  If `True`, ``activeFilterName`` will
            represent the filter of interest and will be used to access the
            appropriate measurements.  If `False`, ``activeFilterName`` will
            represent a stand-in filter that does exist in ``inputCat``, to be
            used as a template table, but all entries will be populates by
            NaNs.

        Returns
        -------
        cat : `pandas.core.frame.DataFrame`
            The collated catalog table including all and only the columns of
            interest for further QA analysis and plotting.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def run(self, inputObjCats, inputObjectTable, tract, band):
        """Collect and aggregate catalog columns required for QA purposes.

        Parameters
        ----------
        inputObjCats : `list` of `pandas.core.frame.DataFrame`
            The input _obj tables from which to collect the relevant data
            columns.
        inputObjectTable : `pandas.core.frame.DataFrame`
            The objectTable_tract associated with the tract.  Only used here to
            ensure that the persisted tables have the same row ordering as the
            objectTable_tract tables for ease of joint use.  As such, only the
            columns attribute actually gets loaded.
        tract : `int`
            The tract for which to aggregate the tables in the requested
            patch list.
        band : `str`
            The band for which to aggregate the input tables.

        Raises
        ------
        RuntimeError
            Raised if config.filterMap was provided, but there is no entry for
            the band requested.
        RuntimeError
            Raised if no catalogs were read in for the given tract.

        Returns
        -------
        allCats : `lsst.pipe.base.Struct`
            Contains the dataFrame with the collated QA table for ``tract`` and
            ``band``.
        """
        # TODO: remove the filterMap config song and dance once DM-28475 lands
        if self.config.filterMap is not None:
            if band in self.config.filterMap:
                filterName = self.config.filterMap[band]
            else:
                raise RuntimeError("A config.filterMap was provided, but there is no entry for the "
                                   "band requested: {}".format(band))
        else:
            filterName = band
        self.log.info("Aggregating QA table for {} tract {} including {} patches".
                      format(filterName, tract, len(inputObjCats)))
        # It is much faster to concatenate a list of DataFrames than to
        # concatenate successively within the for loop.
        catList = []
        loadListObject = None
        for inputCat in inputObjCats:
            inputCatCols = inputCat.get(component="columns")
            inputCatFilters = inputCatCols.get_level_values("filter").unique()
            haveFilter = True
            activeFilterName = filterName
            if filterName not in inputCatFilters:
                haveFilter = False
                activeFilterName = inputCatFilters[0]
            if loadListObject is None:
                loadListObject = self.makeLoadList(inputCatCols, activeFilterName)
            cat = self.collateTable(inputCat, loadListObject, activeFilterName, filterName, haveFilter)
            catList.append(cat)

        self.log.info("Concatenating tables from {} patches".format(len(catList)))
        if not catList:
            raise RuntimeError("No catalogs read for {}.".format(inputObjCats))
        allCats = pd.concat(catList, axis=0)
        # The object "id" is associated with the dataframe index.  Add a
        # column that is the id so that it is available for operations on it,
        # e.g. cat["id"].
        allCats["id"] = allCats.index
        # Reindex to match sorting of objTable_tract tables
        inputObjectTable = inputObjectTable.get(parameters={"columns": []})
        allCats = allCats.reindex(inputObjectTable.index)
        return pipeBase.Struct(qaTractTable=allCats)


class ForcedQaLoadList:
    """Helper class for deriving the column load lists for forced measurements.

    Parameters
    ----------
    inputCatCols : `pandas.core.indexes.multi.MultiIndex`
        A pandas multiIndex object.  The format is assumed to be a list of
        tuples of (dataset, filter, column), from which the list of columns
        associated ``activeFilterName`` is to be isolated as the list under
        consideration.
    actiaveFilterName : `str`
        The name of the filter index key of interest.
    baseColStrList : `list` of `str`
        A list of "prefix" strings of column names from the ``inputCatCols``-
        derived ``columnList``  to include in the return list.  All columns
        that start with one of these strings will be loaded UNLESS the full
        column name contains one of the strings listed in
        ``notInColumnStrList``.
    notInColStrList : `list` of `str`
        A list of substrings to select against when creating return subset list
        of columns from the ``inputCatCols``-derived``columnsList``.
    columnsToCopyFromMeas : `list` of `str`
        A list of string "prefixes" to identify the columns to copy.  All
        columns with names that start with one of these strings will be copied
        from the meas catalogs into the forced_src catalogs UNLESS the full
        column name contains one of the strings listed in
        ``notInColumnStrList``.
    columnsToCopyFromRef : `list` of `str`
        A list of string "prefixes" to identify the columns to copy.  All
        columns with names that start with one of these strings will be copied
        from the ref catalogs into the forced_src catalogs UNLESS the full
        column name contains one of the strings listed in
        ``notInColumnStrList``.

    """
    def __init__(self, inputCatCols, activeFilterName, baseColStrList, notInColStrList,
                 columnsToCopyFromRef, columnsToCopyFromMeas):
        self.forcedCols = deriveColumnsList(inputCatCols, baseColStrList, notInColStrList,
                                            datasetStr="forced_src", activeFilterName=activeFilterName)
        self.refCols = deriveColumnsList(inputCatCols, columnsToCopyFromRef, notInColStrList,
                                         datasetStr="ref", activeFilterName=activeFilterName)
        self.measCols = deriveColumnsList(inputCatCols, columnsToCopyFromMeas, notInColStrList,
                                          datasetStr="meas", activeFilterName=activeFilterName)


class UnforcedQaLoadList:
    """Helper class for deriving the column load list for unforced
    measurements.

    Parameters
    ----------
    inputCatCols : `pandas.core.indexes.multi.MultiIndex`
        A pandas multiIndex object.  The format is assumed to be a list of
        tuples of (dataset, filter, column), from which the list of columns
        associated ``activeFilterName`` is to be isolated as the list under
        consideration.
    actiaveFilterName : `str`
        The name of the filter index key of interest.
    baseColStrList : `list` of `str`
        A list of "prefix" strings of column names from the ``inputCatCols``-
        derived ``columnList``  to include in the return list.  All columns
        that start with one of these strings will be loaded UNLESS the full
        column name contains one of the strings listed in
        ``notInColumnStrList``.
    notInColStrList : `list` of `str`
        A list of substrings to select against when creating return subset list
        of columns from the ``inputCatCols``-derived``columnsList``.
    """
    def __init__(self, inputCatCols, activeFilterName, baseColStrList, notInColStrList):
        self.loadCols = deriveColumnsList(inputCatCols, baseColStrList, notInColStrList, datasetStr="meas",
                                          activeFilterName=activeFilterName)


class MakeForcedQaTractTablesTaskConnections(BaseMakeQaTractTablesTaskConnections,
                                             defaultTemplates={"coaddName": "deep"},
                                             dimensions=("tract", "skymap")):
    pass


class MakeForcedQaTractTablesTaskConfig(BaseMakeQaTractTablesTaskConfig,
                                        pipelineConnections=MakeForcedQaTractTablesTaskConnections):
    # We want the following to come from the *_meas catalogs as they reflect
    # what happened in SFP calibration.
    columnsToCopyFromMeas = pexConfig.ListField(
        dtype=str,
        default=["calib_", ],
        doc=("List of string \"prefixes\" to identify the columns to copy.  All columns with names "
             "that start with one of these strings will be copied from the *_meas catalogs into the "
             "*_forced_src catalogs UNLESS the full column name contains one of the strings listed "
             "in the notInColumnStrList config.")
    )
    # We want the following to come from the *_ref catalogs as they reflect
    # the forced measurement states.
    columnsToCopyFromRef = pexConfig.ListField(
        dtype=str,
        default=["detect_", "merge_peak_sky", "merge_measurement_", ],
        doc=("List of string \"prefixes\" to identify the columns to copy.  All columns with names "
             "that start with one of these strings will be copied from the *_ref catalogs into the "
             "*_forced_src catalogs UNLESS the full column name contains one of the strings listed "
             "in the notInColumnStrList config.")
    )


class MakeForcedQaTractTablesTask(BaseMakeQaTractTablesTask):
    """Read in {coaddName}Coadd_obj tables and aggregate QA columns by tract.

    The values of interest here are those from the forced catalogs (i.e.
    "forced_src") and, in this case, certain columns from the "meas" and "ref"
    catalogs that don't get propagated to the "forced_src" catalogs are added
    to the persisted table, which has dataset type "qaTractTable_forced".

    See BaseMakeQaTractTablesTask for full description.
    """

    ConfigClass = MakeForcedQaTractTablesTaskConfig
    _DefaultName = "makeForcedQaTractTablesTask"

    def makeLoadList(self, inputCatCols, activeFilterName):
        # See the base class definition of this method for a detailed docstring
        forcedLoadInstance = ForcedQaLoadList(inputCatCols, activeFilterName, self.config.baseColStrList,
                                              self.config.notInColStrList, self.config.columnsToCopyFromRef,
                                              self.config.columnsToCopyFromMeas)
        return forcedLoadInstance

    def collateTable(self, inputCat, loadListObject, activeFilterName, filterName, haveFilter):
        # See the base class definition of this method for a detailed docstring
        datasetStr = "forced_src"
        parametersDict = {"columns": {"dataset": datasetStr,
                                      "filter": activeFilterName, "column": loadListObject.forcedCols}}
        forcedCat = inputCat.get(parameters=parametersDict)  # Required because deferLoad=True was set
        forcedCat = forcedCat[datasetStr][activeFilterName]

        # Insert select columns from the ref and meas cats for the forced cats
        datasetStr = "ref"
        parametersDict = {"columns": {"dataset": datasetStr,
                                      "filter": activeFilterName, "column": loadListObject.refCols}}
        refCat = inputCat.get(parameters=parametersDict)  # Required because deferLoad=True was set
        refCat = refCat[datasetStr][activeFilterName]
        datasetStr = "meas"
        parametersDict = {"columns": {"dataset": datasetStr,
                                      "filter": activeFilterName, "column": loadListObject.measCols}}
        measCat = inputCat.get(parameters=parametersDict)  # Required because deferLoad=True was set
        measCat = measCat[datasetStr][activeFilterName]
        if not haveFilter:  # Fill in missing patch columns win NaNs
            patch = inputCat.dataId["patch"]
            # TODO: remove patchId from the log output once it has been
            # switched to the Gen3 sequential integer patch naming
            # convention in the _obj tables.  It is provided now as a
            # convenience while the _obj tables still use the old Gen2
            # "X,Y" notation for the patchId column.
            patchId = forcedCat["patchId"].unique()[0]
            tractId = forcedCat["tractId"].unique()[0]
            self.log.info("Filter {} does not exist for: {}, {:2d} [{}].  Setting NaNs for patch "
                          "columns...".format(filterName, tractId, int(patch), patchId))
            forcedCat = pd.DataFrame().reindex_like(forcedCat)
            refCat = pd.DataFrame().reindex_like(refCat)
            measCat = pd.DataFrame().reindex_like(measCat)
        cat = pd.concat([forcedCat, refCat, measCat], axis=1)
        return cat


class MakeUnforcedQaTractTablesTaskConnections(BaseMakeQaTractTablesTaskConnections,
                                               defaultTemplates={"coaddName": "deep"},
                                               dimensions=("tract", "skymap")):
    pass


class MakeUnforcedQaTractTablesTaskConfig(BaseMakeQaTractTablesTaskConfig,
                                          pipelineConnections=MakeUnforcedQaTractTablesTaskConnections):
    def setDefaults(self):
        super().setDefaults()
        self.connections.qaTractTable = "qaTractTable_unforced"


class MakeUnforcedQaTractTablesTask(BaseMakeQaTractTablesTask):
    """Read in {coaddName}Coadd_obj tables and aggregate QA columns by tract.

    The values of interest here are from just the unforced, "meas", catalogs.

    See BaseMakeQaTractTablesTask for full description.
    """

    ConfigClass = MakeUnforcedQaTractTablesTaskConfig
    _DefaultName = "makeUnforcedQaTractTablesTask"

    def makeLoadList(self, inputCatCols, activeFilterName):
        # See the base class definition of this method for a detailed docstring
        unforcedLoadInstance = UnforcedQaLoadList(inputCatCols, activeFilterName,
                                                  self.config.baseColStrList, self.config.notInColStrList)
        return unforcedLoadInstance

    def collateTable(self, inputCat, loadListObject, activeFilterName, filterName, haveFilter):
        # See the base class definition of this method for a detailed docstring
        datasetStr = "meas"
        parametersDict = {"columns": {"dataset": datasetStr,
                                      "filter": activeFilterName, "column": loadListObject.loadCols}}
        cat = inputCat.get(parameters=parametersDict)  # Required because deferLoad=True was set
        cat = cat[datasetStr][activeFilterName]
        if not haveFilter:  # Fill in missing patch columns win NaNs
            patch = inputCat.dataId["patch"]
            patchId = cat["patchId"].unique()[0]
            tractId = cat["tractId"].unique()[0]
            self.log.info("Filter {} does not exist for: {}, {:2d} [{}].  Setting NaNs for patch "
                          "columns...".format(filterName, tractId, int(patch), patchId))
            cat = pd.DataFrame().reindex_like(cat)
        return cat


def deriveColumnsList(inputCatCols, baseColStrList, notInColStrList, datasetStr=None, activeFilterName=None):
    """Derive a subset list of columns of interest from an existing list.

    The subset list of columns from those in the ``inputCatCols``-derived
    ``columnsList`` that are desired for inclusion in the returned list are
    specified by the the ``baseColStrList`` and ``notInColStrList`` parameters.

    Parameters
    ----------
    inputCatCols : `pandas.core.indexes.multi.MultiIndex` or `list` of `str`
        Either a `list` of `str` representing the column names under
        consideration, or a pandas multiIndex object.  If the latter, the
        format is assumed to be a list of tuples of (dataset, filter, column),
        from which the list of columns associated with ``datasetStr`` and
        ``activeFilterName`` is to be isolated as the list under consideration.
    baseColStrList : `list` of `str`
        A list of "prefix" strings of column names from the ``inputCatCols``-
        derived ``columnList`` to include in the return list.  All columns that
        start with one of these strings will be loaded UNLESS the full column
        name contains one of the strings listed in ``notInColumnStrList``.
    notInColStrList : `list` of `str`
        A list of substrings to select against when creating return subset list
        of columns from the ``inputCatCols``-derived``columnsList``.
    datasetStr : `str` or `None`, optional
        The dataset type of interest.  Can be "forced_src", "ref", or "meas".
        Must be provided for MultiIndex ``inputCatCols`` but is ignored
        otherwise.
    activeFilterName : `str` or `None`, optional
        The name of the filter index key.  Must be provided for MultiIndex
        ``inputCatCols`` but is ignored otherwise.

    Raises
    ------
    RuntimeError
        Raised if unknown type for ``inputCatCols``.
    RuntimeError
        Raised if ``inputCatCols`` is of type
        `pandas.core.indexes.multi.MultiIndex`, but one or both of
        ``datasetStr`` and ``activeFilterName`` are not provided.

    Returns
    -------
    derivedColumnsList : `list` of `str`
       The derived subset list of column names from ``columnsList`` according
       to criteria defined by ``baseColsStrList`` and ``notInColStrList``.
    """
    if isinstance(inputCatCols, pd.core.indexes.multi.MultiIndex):
        columnsList = [tup[2] for tup in inputCatCols if tup[0] == datasetStr and tup[1] == activeFilterName]
        if datasetStr is None or activeFilterName is None:
            raise RuntimeError("Must provide entries for the datasetStr and activeFilterName parameters "
                               "for inputCatCols of type pandas.core.indexes.multi.MultiIndex.")
    elif isinstance(inputCatCols, list):
        columnsList = inputCatCols
    else:
        raise RuntimeError("Unknown type for inputCatCols: {}.  Must be either a list of str or "
                           "pandas.core.indexes.multi.MultiIndex.".format(type(inputCatCols)))
    derivedColumnsList = [column for column in columnsList if (
        column.startswith(tuple(baseColStrList))
        and not any(omitColumn in column for omitColumn in notInColStrList))]
    return derivedColumnsList
