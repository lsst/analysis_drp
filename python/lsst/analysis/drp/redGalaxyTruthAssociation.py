"""Task to associate observed red galaxies with the truth catalog."""

import numpy as np
from smatch.matcher import Matcher
from astropy import units as u
import pandas as pd

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.pex.config.configurableActions import ConfigurableActionStructField

from . import dataSelectors as dataSelectors


class RedGalaxyTruthAssociationConnections(pipeBase.PipelineTaskConnections,
                                           dimensions=("tract", "skymap"),
                                           defaultTemplates={"inputCoaddName": "deep"}):
    object_table = pipeBase.connectionTypes.Input(doc="The tract catalog to do association.",
                                                  storageClass="DataFrame",
                                                  name="objectTable_tract",
                                                  dimensions=("tract", "skymap"),
                                                  deferLoad=True)
    redgal_table = pipeBase.connectionTypes.Input(doc="The red galaxy truth catalog.",
                                                  storageClass="DataFrame",
                                                  name="cosmodc2_1_1_4_redmapper_v0_8_1_redgals",
                                                  deferLoad=True)
    matched_red_galaxies = pipeBase.connectionTypes.Output(doc="Matched object/red galaxy table.",
                                                           storageClass="DataFrame",
                                                           name="matched_true_red_galaxies",
                                                           dimensions=("tract", "skymap"))


class RedGalaxyTruthAssociationConfig(pipeBase.PipelineTaskConfig,
                                      pipelineConnections=RedGalaxyTruthAssociationConnections):
    selector_actions = ConfigurableActionStructField(
        doc="Which selectors to use to narrow down the data for matching.",
        default={"flagSelector": dataSelectors.CoaddPlotFlagSelector},
    )

    bands = pexConfig.ListField(
        doc="Bands to read and store.",
        dtype=str,
        default=["g", "r", "i", "z", "y"]
    )

    flux_fields = pexConfig.ListField(
        doc="Flux fields to use from object table.",
        dtype=str,
        default=["cModelFlux", "gaap0p7Flux", "gaap1p0Flux"]
    )

    err_fields = pexConfig.ListField(
        doc="Error fields to use from object table, matched to flux_fields.",
        dtype=str,
        default=["cModelFluxErr", "gaap0p7FluxErr", "gaap1p0FluxErr"]
    )

    flag_fields = pexConfig.ListField(
        doc="Flag fields to use from object table, matched to flux_fields.",
        dtype=str,
        default=["cModel_flag", "gaapFlux_flag", "gaapFlux_flag"]
    )

    match_radius = pexConfig.Field(
        doc="Radius to match truth galaxies to observed objects (arcsec).",
        dtype=float,
        default=1.0
    )

    def validate(self):
        super().validate()

        if len(self.err_fields) != len(self.flux_fields):
            raise pexConfig.FieldValidationError(RedGalaxyTruthAssociationConfig.err_fields,
                                                 self,
                                                 "err_fields must have same length as flux_fields.")
        if len(self.flag_fields) != len(self.flux_fields):
            raise pexConfig.FieldValidationError(RedGalaxyTruthAssociationConfig.flag_fields,
                                                 self,
                                                 "flag_fields must have same length as flux_fields.")


class RedGalaxyTruthAssociationTask(pipeBase.PipelineTask):

    ConfigClass = RedGalaxyTruthAssociationConfig
    _DefaultName = "redGalaxyTruthAssociationTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        input_ref_dict = butlerQC.get(inputRefs)

        # Determine columns to read
        obj_columns = set({"objectId", "patch", "coord_ra", "coord_dec"})
        for action in self.config.selector_actions:
            for col in action.columns:
                obj_columns.add(col)

        for band in self.config.bands:
            for flux_field, err_field, flag_field in zip(self.config.flux_fields,
                                                         self.config.err_fields,
                                                         self.config.flag_fields):
                obj_columns.update([f"{band}_{flux_field}",
                                    f"{band}_{err_field}",
                                    f"{band}_{flag_field}"])

            obj_columns.add(f"{band}_extendedness")

        redgal_columns = set({"ra", "dec", "ztrue"})
        for band in self.config.bands:
            redgal_columns.add(f"{band}_mag")

        object_table = input_ref_dict["object_table"].get(parameters={"columns": obj_columns})
        redgal_table = input_ref_dict["redgal_table"].get(parameters={"columns": redgal_columns})

        struct = self.run(object_table, redgal_table)

        butlerQC.put(pd.DataFrame(struct.matched_red_galaxies),
                     outputRefs.matched_red_galaxies)

    def run(self, object_table, redgal_table):
        """Run the RedGalaxyTruthAssociationTask.

        Parameters
        ----------
        object_table : `pandas.DataFrame`
            Object table dataframe with select columns.
        redgal_table : `pandas.DataFrame`
            Red galaxy truth table dataframe.

        Returns
        -------
        struct : `lsst.pipe.base.struct`
            Struct with outputs for persistence.
        """
        object_table.reset_index(inplace=True)

        mask = np.ones(len(object_table), dtype=bool)
        for selector in self.config.selector_actions:
            mask &= selector(object_table)

        # Down-select and convert to numpy recarrays
        object_table = object_table[mask].to_records()
        redgal_table = redgal_table.to_records()

        # Match tables
        with Matcher(redgal_table["ra"], redgal_table["dec"]) as matcher:
            idx, i1, i2, d = matcher.query_radius(object_table["coord_ra"],
                                                  object_table["coord_dec"],
                                                  self.config.match_radius/3600.,
                                                  return_indices=True)
        # Output dtype
        dtype = [("objectId", "i8"),
                 ("coord_ra", "f8"),
                 ("coord_dec", "f8"),
                 ("patch", "i4")]

        obj_cols_to_copy = set()
        for band in self.config.bands:
            for flux_field in self.config.flux_fields:
                flux_col = f"{band}_{flux_field}"
                if flux_col not in obj_cols_to_copy:
                    dtype.append((flux_col, "f4"))
                    obj_cols_to_copy.add(flux_col)
            for err_field in self.config.err_fields:
                err_col = f"{band}_{err_field}"
                if err_col not in obj_cols_to_copy:
                    dtype.append((err_col, "f4"))
                    obj_cols_to_copy.add(err_col)
            for flag_field in self.config.flag_fields:
                flag_col = f"{band}_{flag_field}"
                if flag_col not in obj_cols_to_copy:
                    dtype.append((flag_col, "?"))
                    obj_cols_to_copy.add(flag_col)

            dtype.append((f"{band}_trueFlux", "f4"))

            extendedness_col = f"{band}_extendedness"
            dtype.append((extendedness_col, "f4"))
            obj_cols_to_copy.add(extendedness_col)

        matched_red_galaxies = np.zeros(i1.size, dtype=dtype)
        matched_red_galaxies["coord_ra"] = redgal_table["ra"][i1]
        matched_red_galaxies["coord_dec"] = redgal_table["dec"][i1]
        matched_red_galaxies["objectId"] = object_table["objectId"][i2]
        matched_red_galaxies["patch"] = object_table["patch"][i2]

        for obj_col_to_copy in obj_cols_to_copy:
            matched_red_galaxies[obj_col_to_copy] = object_table[obj_col_to_copy][i2]

        for band in self.config.bands:
            mag = redgal_table[f"{band}_mag"]*u.ABmag
            matched_red_galaxies[f"{band}_trueFlux"] = mag.to(u.nJy).value[i1]

        return pipeBase.Struct(matched_red_galaxies=matched_red_galaxies)
