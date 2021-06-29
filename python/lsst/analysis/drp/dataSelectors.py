__all__ = ("HighSnSelector", "LowSnSelector", "MainFlagSelector", "PsfFlagSelector", "StarIdentifier",
           "GalaxyIdentifier", "UnknownIdentifier", "CoaddPlotFlagSelector")

from lsst.pex.config import ListField, Field
from lsst.pipe.tasks.dataFrameActions import DataFrameAction
import numpy as np


class FlagSelector(DataFrameAction):
    """The base flag selector to use to select valid sources for QA"""

    selectWhenFalse = ListField(doc="Names of the flag columns to select on when False",
                                dtype=str,
                                optional=False,
                                default=[])

    selectWhenTrue = ListField(doc="Names of the flag columns to select on when False",
                               dtype=str,
                               optional=False,
                               default=[])

    @property
    def columns(self):
        allCols = list(self.selectWhenFalse) + list(self.selectWhenTrue)
        yield from allCols

    def __call__(self, df, **kwargs):
        """Select on the given flags"""
        result = None
        for flag in self.selectWhenFalse:
            selected = (df[flag] == 0)
            if result is None:
                result = selected
            else:
                result &= selected
        for flag in self.selectWhenTrue:
            selected = (df[flag] == 1)
            if result is None:
                result = selected
            else:
                result &= selected
        return result


class MainFlagSelector(FlagSelector):
    """The main set of flags to use to select valid sources for QA"""

    bands = ListField(doc="The bands to apply the flags in",
                      dtype=str,
                      default=["g", "r", "i", "z", "y"])

    def setDefaults(self):
        super().setDefaults()
        flagCols = ["PixelFlags", "Blendedness_flag"]
        filterColumns = [band + flag for flag in flagCols for band in self.bands]
        self.selectWhenFalse = filterColumns


class PsfFlagSelector(FlagSelector):
    """Remove sources with bad flags set for CModel measurements"""

    bands = ListField(doc="The bands to apply the flags in",
                      dtype=str,
                      default=["g", "r", "i", "z", "y"])

    def setDefaults(self):
        super().setDefaults()
        flagCols = ["PsfFlux_flag", "PsfFlux_flag_apCorr", "PsfFlux_flag_edge", "PsfFlux_flag_noGoodPixels"]
        filterColumns = [band + flag for flag in flagCols for band in self.bands]
        self.selectWhenFalse = filterColumns


class BaseSNRSelector(DataFrameAction):
    fluxField = Field(doc="Flux field to use in SNR calculation", dtype=str,
                      default="PsFlux", optional=False)
    errField = Field(doc="Flux err field to use in SNR calculation", dtype=str,
                     default="PsFluxErr", optional=False)
    threshold = Field(doc="The signal to noise threshold to select sources",
                      dtype=float,
                      optional=False)
    band = Field(doc="The band to make the selection in.",
                 default="i",
                 dtype=str)

    @property
    def columns(self):
        return (self.band + self.fluxField, self.band + self.errField)


class SnSelector(DataFrameAction):
    """Selects points that have S/N > threshold in the given flux type"""
    fluxType = Field(doc="Flux type to calculate the S/N in.", dtype=str, default="iPsFlux")
    threshold = Field(doc="The S/N threshold to remove sources with.", dtype=float, default=500.0)

    @property
    def columns(self):
        return (self.fluxType, self.fluxType + "Err")

    def __call__(self, df):
        return (df[self.fluxType] / df[self.fluxType + "Err"]) > self.threshold


class HighSnSelector(BaseSNRSelector):
    """Select high signal to noise sources, in PSF flux"""

    def __call__(self, df, **kwargs):
        """Selects sources with PSF SN ratio above self.threshold"""

        return (df[self.band + self.fluxField] / df[self.band + self.errField]) > self.threshold

    def setDefaults(self):
        super().setDefaults()
        self.threshold = 2700


class LowSnSelector(BaseSNRSelector):
    """Select lower signal to noise sources, in PSF flux"""

    def __call__(self, df, **kwargs):
        """Selects sources with PSF SN ratio below self.threshold"""

        return (df[self.band + self.fluxField] / df[self.band + self.errField]) > self.threshold

    def setDefaults(self):
        super().setDefaults()
        self.threshold = 500


class StarIdentifier(DataFrameAction):
    """Identifies stars from the dataFrame and marks them as a 1
       in the added sourceType column"""

    band = Field(doc="The band the object is to be classified as a star in.",
                 default="i",
                 dtype=str)

    @property
    def columns(self):
        return [self.band + "Extendedness"]

    def __call__(self, df, **kwargs):
        """Identidifies sources classed as stars"""
        stars = (df[self.band + "Extendedness"] == 0.0)
        sourceType = np.zeros(len(df))
        sourceType[stars] = 1
        return sourceType


class GalaxyIdentifier(DataFrameAction):
    """Identifies galaxies from the dataFrame and marks them as a 2
       in the added sourceType column"""

    band = Field(doc="The band the object is to be classified as a galaxy in.",
                 default="i",
                 dtype=str)

    @property
    def columns(self):
        return [self.band + "Extendedness"]

    def __call__(self, df, **kwargs):
        """Identifies sources classed as galaxies"""
        gals = (df[self.band + "Extendedness"] == 1.0)
        sourceType = np.zeros(len(df))
        sourceType[gals] = 2
        return sourceType


class UnknownIdentifier(DataFrameAction):
    """Identifies un classified objects from the dataFrame and marks them as a
       9 in the added sourceType column"""

    band = Field(doc="The band the object is to be classified as an unkown in.",
                 default="i",
                 dtype=str)

    @property
    def columns(self):
        return [self.band + "Extendedness"]

    def __call__(self, df, **kwargs):
        """Identifies sources classed as unknowns"""
        unknowns = (df[self.band + "Extendedness"] == 9.0)
        sourceType = np.zeros(len(df))
        sourceType[unknowns] = 9
        return sourceType


class CoaddPlotFlagSelector(FlagSelector):
    """The flags to use for selecting sources for coadd QA"""

    bands = ListField(doc="The bands to apply the flags in",
                      dtype=str,
                      default=["g", "r", "i", "z", "y"])

    def setDefaults(self):
        super().setDefaults()
        # This is missing one of the PA flags because they are not
        # currently in the objectTablesbase_SdssCentroid_flag, is it
        # redundant with xy_flag?
        flagCols = ["PsfFlux_flag", "PixelFlags_saturatedCenter", "Extendedness_flag"]

        filterColumns = ["xy_flag"] + [band + flag for flag in flagCols for band in self.bands]
        self.selectWhenFalse = filterColumns
