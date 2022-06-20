import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import lsst.geom
from lsst.meas.algorithms import ReferenceObjectLoader, LoadReferenceObjectsTask
from lsst.pipe.tasks.configurableActions import ConfigurableActionStructField
from lsst.skymap import BaseSkyMap

from . import dataSelectors

__all__ = ["CatalogMatchConfig", "CatalogMatchTask", "AstropyMatchConfig", "AstropyMatchTask"]


class AstropyMatchConfig(pexConfig.Config):

    maxDistance = pexConfig.Field(
        doc="Max distance between matches in arcsec",
        dtype=float,
        default=1.0,
    )


class AstropyMatchTask(pipeBase.Task):
    """ A task for running the astropy matcher `match_to_catalog_sky` on
    between target and reference catalogs."""

    ConfigClass = AstropyMatchConfig

    def run(self, refCatalog, targetCatalog):
        """Run matcher

        Parameters
        ----------
        refCatalog: `pandas.core.frame.DataFrame`
            The reference catalog with coordinates in degrees
        targetCatalog: `pandas.core.frame.DataFrame`
            The target catalog with coordinates in degrees

        Returns
        -------
        `pipeBase.Struct` containing:
            refMatchIndices: `numpy.ndarray`
                Array of indices of matched reference catalog objects
            targetMatchIndices: `numpy.ndarray`
                Array of indices of matched target catalog objects
            separations: `astropy.coordinates.angles.Angle`
                Array of angle separations between matched objects
        """
        refCat_ap = SkyCoord(ra=refCatalog['coord_ra'] * u.degree,
                             dec=refCatalog['coord_dec'] * u.degree)

        sourceCat_ap = SkyCoord(ra=targetCatalog['coord_ra'] * u.degree,
                                dec=targetCatalog['coord_dec'] * u.degree)

        id, d2d, d3d = refCat_ap.match_to_catalog_sky(sourceCat_ap)

        goodMatches = d2d.arcsecond < self.config.maxDistance

        refMatchIndices = np.flatnonzero(goodMatches)
        targetMatchIndices = id[goodMatches]

        separations = d2d[goodMatches].arcsec

        return pipeBase.Struct(refMatchIndices=refMatchIndices,
                               targetMatchIndices=targetMatchIndices,
                               separations=separations)


class CatalogMatchConnections(pipeBase.PipelineTaskConnections, dimensions=("tract", "skymap"),
                              defaultTemplates={"targetCatalog": "objectTable_tract",
                                                "refCatalog": "astrometryRefCat"}):

    catalog = pipeBase.connectionTypes.Input(
        doc="The tract-wide catalog to make plots from.",
        storageClass="DataFrame",
        name="{targetCatalog}",
        dimensions=("tract", "skymap"),
        deferLoad=True
    )

    refCat = pipeBase.connectionTypes.PrerequisiteInput(
        doc="The reference catalog to match to loaded input catalog sources.",
        name="gaia_dr2_20200414",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        deferLoad=True,
        multiple=True
    )

    skymap = pipeBase.connectionTypes.Input(
        doc="The skymap for the tract",
        storageClass="SkyMap",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        dimensions=("skymap",)
    )

    matchCatalog = pipeBase.connectionTypes.Output(
        doc="Catalog with matched target and reference objects with separations",
        name="{targetCatalog}_{refCatalog}_match",
        storageClass="DataFrame",
        dimensions=("tract", "skymap")
    )


class CatalogMatchConfig(pipeBase.PipelineTaskConfig, pipelineConnections=CatalogMatchConnections):

    astrometryRefObjLoader = pexConfig.ConfigurableField(
        target=LoadReferenceObjectsTask,
        doc="Reference object loader for astrometric fit",
    )

    matcher = pexConfig.ConfigurableField(
        target=AstropyMatchTask,
        doc="Task for matching refCat and SourceCatalog"
    )

    epoch = pexConfig.Field(
        doc="Epoch to which reference objects are shifted",
        dtype=float,
        default=2015.0
    )

    selectorActions = ConfigurableActionStructField(
        doc="Which selectors to use to narrow down the data for QA plotting.",
        default={"flagSelector": dataSelectors.CoaddPlotFlagSelector},
    )

    sourceSelectorActions = ConfigurableActionStructField(
        doc="What types of sources to use.",
        default={"sourceSelector": dataSelectors.StarIdentifier},
    )

    extraColumnSelectors = ConfigurableActionStructField(
        doc="Other selectors that are not used in this task, but whose columns"
            "may be needed downstream",
        default={"selector1": dataSelectors.SnSelector,
                 "selector2": dataSelectors.GalaxyIdentifier}
    )

    extraColumns = pexConfig.ListField(
        doc="Other catalog columns to persist to downstream tasks",
        dtype=str,
        default=['i_cModelFlux', 'x', 'y']
    )

    def setDefaults(self):
        self.astrometryRefObjLoader.requireProperMotion = False
        self.astrometryRefObjLoader.anyFilterMapsToThis = 'phot_g_mean'
        for selectorActions in [self.selectorActions, self.sourceSelectorActions,
                                self.extraColumnSelectors]:
            for selector in selectorActions:
                if 'bands' in selector.names():
                    selector.bands = ["g", "r", "i", "z", "y"]
                elif 'band' in selector.names():
                    selector.band = 'i'


class CatalogMatchTask(pipeBase.PipelineTask):
    """Match a tract-level catalog to a reference catalog
    """

    ConfigClass = CatalogMatchConfig
    _DefaultName = "catalogMatch"

    def __init__(self, butler=None, initInputs=None, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("matcher")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class

        inputs = butlerQC.get(inputRefs)

        columns = ['coord_ra', 'coord_dec', 'patch'] + self.config.extraColumns.list()
        for selectorAction in [self.config.selectorActions, self.config.sourceSelectorActions,
                               self.config.extraColumnSelectors]:
            for selector in selectorAction:
                columns += list(selector.columns)

        dataFrame = inputs["catalog"].get(parameters={"columns": columns})
        inputs['catalog'] = dataFrame

        tract = butlerQC.quantum.dataId['tract']

        self.refObjLoader = ReferenceObjectLoader(dataIds=[ref.datasetRef.dataId
                                                           for ref in inputRefs.refCat],
                                                  refCats=inputs.pop('refCat'),
                                                  config=self.config.astrometryRefObjLoader,
                                                  log=self.log)

        self.setRefCat(inputs.pop('skymap'), tract)

        outputs = self.run(**inputs)

        butlerQC.put(outputs, outputRefs)

    def run(self, catalog):
        """Prep the catalog and run the matcher.

        Parameters
        ----------
        catalog : `pandas.core.frame.DataFrame`

        Returns
        -------
        `pipeBase.Struct` containing:
            matchedCat : `pandas.core.frame.DataFrame`
                Catalog containing the matched objects with all columns from
                the original input catalogs, with the suffix "_ref" or
                "_target" for duplicated column names, plus a column with the
                angular separation in arcseconds between matches.
        """
        # Apply the selectors to the catalog
        mask = np.ones(len(catalog), dtype=bool)
        for selector in self.config.selectorActions:
            mask &= selector(catalog)

        for selector in self.config.sourceSelectorActions:
            mask &= selector(catalog).astype(bool)
        targetCatalog = catalog[mask]
        targetCatalog = targetCatalog.reset_index()

        if (len(targetCatalog) == 0) or (len(self.refCat) == 0):
            matches = pipeBase.Struct(refMatchIndices=np.array([]),
                                      targetMatchIndices=np.array([]),
                                      separations=np.array([]))
        else:
            # Run the matcher
            matches = self.matcher.run(self.refCat, targetCatalog)

        # Join the catalogs for the matched catalogs
        refMatches = self.refCat.iloc[matches.refMatchIndices].reset_index()
        sourceMatches = targetCatalog.iloc[matches.targetMatchIndices].reset_index()
        matchedCat = sourceMatches.join(refMatches, lsuffix='_target', rsuffix='_ref')

        separations = pd.Series(matches.separations).rename("separation")
        matchedCat = matchedCat.join(separations)

        return pipeBase.Struct(matchCatalog=matchedCat)

    def setRefCat(self, skymap, tract):
        """Make a reference catalog with coordinates in degrees

        Parameters
        ----------
        skymap : `lsst.skymap`
            The skymap used to define the patch boundaries.
        tract : int
            The tract corresponding to the catalog data.
        """
        # Load the reference objects in a skyCircle around the tract
        tractInfo = skymap.generateTract(tract)
        boundingCircle = tractInfo.getOuterSkyPolygon().getBoundingCircle()
        center = lsst.geom.SpherePoint(boundingCircle.getCenter())
        radius = boundingCircle.getOpeningAngle()

        epoch = Time(self.config.epoch, format="decimalyear")

        skyCircle = self.refObjLoader.loadSkyCircle(center,
                                                    radius,
                                                    'i',
                                                    epoch=epoch)
        refCat = skyCircle.refCat

        # Convert the coordinates to RA/Dec and convert the catalog to a
        # dataframe
        refCat['coord_ra'] = (refCat['coord_ra'] * u.radian).to(u.degree).to_value()
        refCat['coord_dec'] = (refCat['coord_dec'] * u.radian).to(u.degree).to_value()
        self.refCat = refCat.asAstropy().to_pandas()


class CatalogMatchVisitConnections(pipeBase.PipelineTaskConnections, dimensions=("visit", "skymap"),
                                   defaultTemplates={"targetCatalog": "sourceTable_visit",
                                                     "refCatalog": "astrometryRefCat"}):

    catalog = pipeBase.connectionTypes.Input(
        doc="The visit-wide catalog to make plots from.",
        storageClass="DataFrame",
        name="sourceTable_visit",
        dimensions=("visit",),
        deferLoad=True
    )

    refCat = pipeBase.connectionTypes.PrerequisiteInput(
        doc="The astrometry reference catalog to match to loaded input catalog sources.",
        name="gaia_dr2_20200414",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        deferLoad=True,
        multiple=True
    )

    visitSummaryTable = pipeBase.connectionTypes.Input(
        doc="A summary table of the ccds in the visit",
        storageClass="ExposureCatalog",
        name="visitSummary",
        dimensions=("visit",)
    )

    matchCatalog = pipeBase.connectionTypes.Output(
        doc="Catalog with matched target and reference objects with separations",
        name="{targetCatalog}_{refCatalog}_match",
        storageClass="DataFrame",
        dimensions=("visit",)
    )


class CatalogMatchVisitConfig(CatalogMatchTask.ConfigClass,
                              pipelineConnections=CatalogMatchVisitConnections):

    extraColumns = pexConfig.ListField(
        doc="Other catalog columns to persist to downstream tasks",
        dtype=str,
        default=["psfFlux", "psfFluxErr"]
    )

    def setDefaults(self):
        self.astrometryRefObjLoader.requireProperMotion = False
        self.astrometryRefObjLoader.anyFilterMapsToThis = 'phot_g_mean'
        for selectorActions in [self.selectorActions, self.sourceSelectorActions,
                                self.extraColumnSelectors]:
            for selector in selectorActions:
                if 'bands' in selector.names():
                    selector.bands = []
                elif 'band' in selector.names():
                    selector.band = ""


class CatalogMatchVisitTask(CatalogMatchTask):
    """Match a visit-level catalog to a reference catalog
    """

    ConfigClass = CatalogMatchVisitConfig
    _DefaultName = "catalogMatchVisit"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class

        inputs = butlerQC.get(inputRefs)

        columns = ['coord_ra', 'coord_dec', 'detector'] + self.config.extraColumns.list()
        for selectorAction in [self.config.selectorActions, self.config.sourceSelectorActions,
                               self.config.extraColumnSelectors]:
            for selector in selectorAction:
                columns += list(selector.columns)

        dataFrame = inputs["catalog"].get(parameters={"columns": columns})
        inputs['catalog'] = dataFrame

        self.refObjLoader = ReferenceObjectLoader(dataIds=[ref.datasetRef.dataId
                                                           for ref in inputRefs.refCat],
                                                  refCats=inputs.pop('refCat'),
                                                  config=self.config.astrometryRefObjLoader,
                                                  log=self.log)

        self.setRefCat(inputs.pop('visitSummaryTable'))

        outputs = self.run(**inputs)

        butlerQC.put(outputs, outputRefs)

    def setRefCat(self, visitSummaryTable):
        """Make a reference catalog with coordinates in degrees

        Parameters
        ----------
        visitSummaryTable : `lsst.afw.table.ExposureCatalog`
            The table of visit information
        """
        # Get convex hull around the detectors, then get its center and radius
        corners = []
        for visSum in visitSummaryTable:
            for (ra, dec) in zip(visSum['raCorners'], visSum['decCorners']):
                corners.append(lsst.geom.SpherePoint(ra, dec, units=lsst.geom.degrees).getVector())
        visitBoundingCircle = lsst.sphgeom.ConvexPolygon.convexHull(corners).getBoundingCircle()
        center = lsst.geom.SpherePoint(visitBoundingCircle.getCenter())
        radius = visitBoundingCircle.getOpeningAngle()

        # Get the observation date of the visit
        obsDate = visSum.getVisitInfo().getDate()
        epoch = Time(obsDate.toPython())

        # Load the reference catalog in the skyCircle of the detectors, then
        # convert the coordinates to degrees and convert the catalog to a
        # dataframe
        skyCircle = self.refObjLoader.loadSkyCircle(center,
                                                    radius,
                                                    'i',
                                                    epoch=epoch)
        refCat = skyCircle.refCat

        refCat['coord_ra'] = (refCat['coord_ra'] * u.radian).to(u.degree).to_value()
        refCat['coord_dec'] = (refCat['coord_dec'] * u.radian).to(u.degree).to_value()
        self.refCat = refCat.asAstropy().to_pandas()
