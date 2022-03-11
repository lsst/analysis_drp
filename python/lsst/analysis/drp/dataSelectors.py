__all__ = ("FlagSelector", "PsfFlagSelector", "StarIdentifier", "GalaxyIdentifier", "UnknownIdentifier",
           "CoaddPlotFlagSelector", "BaseSNRSelector", "SnSelector")

from lsst.pex.config import ListField, Field
from lsst.pipe.tasks.dataFrameActions import DataFrameAction
import numpy as np


class FlagSelector(DataFrameAction):
    """The base flag selector to use to select valid sources for QA"""

    selectWhenFalse = ListField(doc="Names of the flag columns to select on when False",
                                dtype=str,
                                optional=False,
                                default=[])

    selectWhenTrue = ListField(doc="Names of the flag columns to select on when True",
                               dtype=str,
                               optional=False,
                               default=[])

    @property
    def columns(self):
        allCols = list(self.selectWhenFalse) + list(self.selectWhenTrue)
        yield from allCols

    def __call__(self, df, **kwargs):
        """Select on the given flags

        Parameters
        ----------
        df : `pandas.core.frame.DataFrame`

        Returns
        -------
        result : `numpy.ndarray`
            A mask of the objects that satisfy the given
            flag cuts.

        Notes
        -----
        Uses the columns in selectWhenFalse and
        selectWhenTrue to decide which columns to
        select on in each circumstance.
        """
        result = np.ones(len(df), dtype=bool)
        for flag in self.selectWhenFalse:
            result &= (df[flag].values == 0)
        for flag in self.selectWhenTrue:
            result &= (df[flag].values == 1)
        return result


class PsfFlagSelector(FlagSelector):
    """Remove sources with bad flags set for PSF measurements """

    bands = ListField(doc="The bands to apply the flags in",
                      dtype=str,
                      default=["g", "r", "i", "z", "y"])

    @property
    def columns(self):
        filterColumns = ["xy_flag", "detect_isPatchInner", "detect_isDebelendedSource"]
        flagCols = ["psfFlux_flag", "psfFlux_flag_apCorr", "psfFlux_flag_edge",
                    "psfFlux_flag_noGoodPixels"]
        filterColumns += [band + "_" + flag if len(band) > 0 else band + flag
                          for flag in flagCols for band in self.bands]
        yield from filterColumns

    def __call__(self, df, **kwargs):
        """Make a mask of objects with bad PSF flags

        Parameters
        ----------
        df : `pandas.core.frame.DataFrame`

        Returns
        -------
        result : `numpy.ndarray`
            A mask of the objects that satisfy the given
            flag cuts.

        Notes
        -----
        Uses the PSF flags and some general quality
        control flags to make a mask of the data that
        satisfies these criteria.
        """

        result = None
        flagCols = ["psfFlux_flag", "pixelFlags_saturatedCenter", "extendedness_flag"]
        filterColumns = ["xy_flag"]
        filterColumns += [band + "_" + flag if len(band) > 0 else band + flag
                          for flag in flagCols for band in self.bands]
        for flag in filterColumns:
            selected = (df[flag].values == 0)
            if result is None:
                result = selected
            else:
                result &= selected
        result &= (df["detect_isPatchInner"] == 1)
        result &= (df["detect_isDeblendedSource"] == 1)
        return result


class BaseSNRSelector(DataFrameAction):
    """Selects sources that have a S/N greater than the
    given threshold"""

    fluxField = Field(doc="Flux field to use in SNR calculation", dtype=str,
                      default="psfFlux", optional=False)
    errField = Field(doc="Flux err field to use in SNR calculation", dtype=str,
                     default="psfFluxErr", optional=False)
    threshold = Field(doc="The signal to noise threshold to select sources",
                      dtype=float,
                      optional=False)
    band = Field(doc="The band to make the selection in.",
                 default="i",
                 dtype=str)

    @property
    def columns(self):
        if len(self.band) > 0:
            band = self.band + "_"
        else:
            band = self.band
        return (band + self.fluxField, band + self.errField)


class SnSelector(DataFrameAction):
    """Selects points that have S/N > threshold in the given flux type"""
    fluxType = Field(doc="Flux type to calculate the S/N in.",
                     dtype=str,
                     default="psfFlux")
    threshold = Field(doc="The S/N threshold to remove sources with.",
                      dtype=float,
                      default=500.0)
    bands = ListField(doc="The bands to apply the signal to noise cut in.",
                      dtype=str,
                      default=["i"])

    @property
    def columns(self):
        cols = []
        for band in self.bands:
            if len(band) > 0:
                band = band + "_"
            cols += [band + self.fluxType, band + self.fluxType + "Err"]

        return cols

    def __call__(self, df):
        """Makes a mask of objects that have S/N greater than
        self.threshold in self.fluxType

        Parameters
        ----------
        df : `pandas.core.frame.DataFrame`

        Returns
        -------
        result : `numpy.ndarray`
            A mask of the objects that satisfy the given
            S/N cut.
        """

        mask = np.ones(len(df), dtype=bool)
        for band in self.bands:
            if len(band) > 0:
                band = band + "_"
            mask &= ((df[band + self.fluxType] / df[band + self.fluxType + "Err"]) > self.threshold)

        return mask


class StarIdentifier(DataFrameAction):
    """Identifies stars from the dataFrame and marks them as a 1
       in the added sourceType column"""

    band = Field(doc="The band the object is to be classified as a star in.",
                 default="i",
                 dtype=str)

    @property
    def columns(self):
        if len(self.band) > 0:
            band = self.band + "_"
        else:
            band = self.band
        return [band + "extendedness"]

    def __call__(self, df, **kwargs):
        """Identifies sources classified as stars

        Parameters
        ----------
        df : `pandas.core.frame.DataFrame`

        Returns
        -------
        result : `numpy.ndarray`
            An array with the objects that are classified as
            stars marked with a 1.
        """
        if len(self.band) > 0:
            band = self.band + "_"
        else:
            band = self.band
        stars = (df[band + "extendedness"] == 0.0)
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
        if len(self.band) > 0:
            band = self.band + "_"
        else:
            band = self.band
        return [band + "extendedness"]

    def __call__(self, df, **kwargs):
        """Identifies sources classified as galaxies

        Parameters
        ----------
        df : `pandas.core.frame.DataFrame`

        Returns
        -------
        result : `numpy.ndarray`
            An array with the objects that are classified as
            galaxies marked with a 2.
        """
        if len(self.band) > 0:
            band = self.band + "_"
        else:
            band = self.band
        gals = (df[band + "extendedness"] == 1.0)
        sourceType = np.zeros(len(df))
        sourceType[gals] = 2
        return sourceType


class UnknownIdentifier(DataFrameAction):
    """Identifies un classified objects from the dataFrame and marks them as a
       9 in the added sourceType column"""

    band = Field(doc="The band the object is to be classified as an unknown in.",
                 default="i",
                 dtype=str)

    @property
    def columns(self):
        if len(self.band) > 0:
            band = self.band + "_"
        else:
            band = self.band
        return [band + "extendedness"]

    def __call__(self, df, **kwargs):
        """Identifies sources classed as unknowns

        Parameters
        ----------
        df : `pandas.core.frame.DataFrame`

        Returns
        -------
        result : `numpy.ndarray`
            An array with the objects that are classified as
            unknown marked with a 9.
        """
        if len(self.band) > 0:
            band = self.band + "_"
        else:
            band = self.band
        unknowns = (df[band + "extendedness"] == 9.0)
        sourceType = np.zeros(len(df))
        sourceType[unknowns] = 9
        return sourceType


class VisitPlotFlagSelector(DataFrameAction):

    @property
    def columns(self):
        flagCols = ["psfFlux_flag", "pixelFlags_saturatedCenter", "extendedness_flag", "centroid_flag"]
        yield from flagCols

    def __call__(self, df, **kwargs):
        """The flags to use for selecting sources for visit QA

        Parameters
        ----------
        df : `pandas.core.frame.DataFrame`

        Returns
        -------
        result : `numpy.ndarray`
            A mask of the objects that satisfy the given
            flag cuts.

        Notes
        -----
        These flags are taken from pipe_analysis and are considered to
        be the standard flags for general QA plots. Some of the plots
        will require a different set of flags, or additional ones on
        top of the ones specified here. These should be specifed in
        an additional selector rather than adding to this one.
        """

        result = None
        flagCols = ["psfFlux_flag", "pixelFlags_saturatedCenter", "extendedness_flag", "centroid_flag"]
        for flag in flagCols:
            selected = (df[flag].values == 0)
            if result is None:
                result = selected
            else:
                result &= selected
        return result


class CoaddPlotFlagSelector(DataFrameAction):
    """The flags to use for selecting sources for coadd QA

    Parameters
    ----------
    df : `pandas.core.frame.DataFrame`

    Returns
    -------
    result : `numpy.ndarray`
        A mask of the objects that satisfy the given
        flag cuts.

    Notes
    -----
    These flags are taken from pipe_analysis and are considered to
    be the standard flags for general QA plots. Some of the plots
    will require a different set of flags, or additional ones on
    top of the ones specified here. These should be specifed in
    an additional selector rather than adding to this one.
    """

    bands = ListField(doc="The bands to apply the flags in",
                      dtype=str,
                      default=["g", "r", "i", "z", "y"])

    @property
    def columns(self):
        flagCols = ["psfFlux_flag", "pixelFlags_saturatedCenter", "extendedness_flag"]
        filterColumns = ["xy_flag", "detect_isPatchInner", "detect_isDeblendedSource"]
        filterColumns += [band + "_" + flag if len(band) > 0 else band + flag
                          for flag in flagCols for band in self.bands]
        yield from filterColumns

    def __call__(self, df, **kwargs):
        """The flags to use for selecting sources for coadd QA

        Parameters
        ----------
        df : `pandas.core.frame.DataFrame`

        Returns
        -------
        result : `numpy.ndarray`
            A mask of the objects that satisfy the given
            flag cuts.

        Notes
        -----
        These flags are taken from pipe_analysis and are considered to
        be the standard flags for general QA plots. Some of the plots
        will require a different set of flags, or additional ones on
        top of the ones specified here. These should be specifed in
        an additional selector rather than adding to this one.
        """

        result = None
        flagCols = ["psfFlux_flag", "pixelFlags_saturatedCenter", "extendedness_flag"]
        filterColumns = ["xy_flag"]
        filterColumns += [band + "_" + flag if len(band) > 0 else band + flag
                          for flag in flagCols for band in self.bands]
        for flag in filterColumns:
            selected = (df[flag].values == 0)
            if result is None:
                result = selected
            else:
                result &= selected
        result &= (df["detect_isPatchInner"] == 1)
        result &= (df["detect_isDeblendedSource"] == 1)
        return result
