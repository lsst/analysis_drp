from __future__ import annotations

import lsst.pipe.base as pipeBase

from .qaUtils import (addMagnitudes, addColors, stellarLocusFits, addUseForQAFlag, addUseForStatsColumn,
                      addSNColumn, addDeconvMoments)


class AddQAColumnsTaskConnections(pipeBase.PipelineTaskConnections, dimensions=("tract", "skymap")):

    cat = pipeBase.connectionTypes.Input(doc="The tract wide catalog to make plots from.",
                                         storageClass="DataFrame",
                                         name="objectTable_tract",
                                         dimensions=("tract", "skymap"))

    qaCat = pipeBase.connectionTypes.Output(doc="The input catalog with additional columns for QA added",
                                            storageClass="DataFrame",
                                            name="qaTable_tract",
                                            dimensions=("tract", "skymap"))


class AddQAColumnsTaskConfig(pipeBase.PipelineTaskConfig, pipelineConnections=AddQAColumnsTaskConnections):

    pass


class AddQAColumnsTask(pipeBase.PipelineTask):

    ConfigClass = AddQAColumnsTaskConfig
    _DefaultName = "addQAColumnsTask"

    def run(self, cat):
        """Add the columns required for QA purposes

        Parameters
        ----------
        cat : `pandas.core.frame.DataFrame`

        Returns
        -------
        pipeBase.Struct
            Contains the dataFrame with the new columns added

        Notes
        -----
        See qaUtils for documentation on individual functions
        """
        # TODO: Get rid of warnings
        # TODO: raise exceptions if needed e.g. all nan column

        self.log.info("Adding additional QA columns to the object table")
        cat = addMagnitudes(cat)
        cat = addMagnitudes(cat, fluxColName="CModelFlux")
        cat = addColors(cat)
        cat = addColors(cat, magColName="CModelMag")
        cat = stellarLocusFits(cat)
        cat = stellarLocusFits(cat, magColName="CModelMag")
        cat = addSNColumn(cat)
        cat = addUseForQAFlag(cat)
        cat = addUseForStatsColumn(cat)

        # TODO: Maybe split into different functions for each pipeline
        cat = addDeconvMoments(cat)

        return pipeBase.Struct(qaCat=cat)
