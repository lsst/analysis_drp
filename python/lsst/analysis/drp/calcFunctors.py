__all__ = ["SNCalculator", "KronFluxDivPsfFlux", "MagDiff", "ColorDiff"]

from lsst.pipe.tasks.dataFrameActions import DivideColumns, MultiColumnAction
from lsst.pex.config import Field
from astropy import units as u
import numpy as np


class SNCalculator(DivideColumns):
    """Calculate the signal to noise by default the i band PSF flux is used"""

    def setDefaults(self):
        super().setDefaults()
        self.colA.column = "i_psfFlux"
        self.colB.column = "i_psfFluxErr"


class KronFluxDivPsfFlux(DivideColumns):
    """Divide the Kron instFlux by the PSF instFlux"""

    def setDefaults(self):
        super().setDefaults()
        self.colA.column = "i_kronFlux"
        self.colB.column = "i_psfFlux"


class MagDiff(MultiColumnAction):
    """Calculate the difference between two magnitudes;
    each magnitude is derived from a flux column.

    Parameters
    ----------
    df : `pandas.core.frame.DataFrame`
        The catalog to calculate the magnitude difference from.

    Returns
    -------
    The magnitude difference in milli mags.

    Notes
    -----
    The flux columns need to be in units (specifiable in
    the fluxUnits1 and 2 config options) that can be converted
    to janskies. This action doesn't have any calibration
    information and assumes that the fluxes are already
    calibrated.
    """

    col1 = Field(doc="Column to subtract from", dtype=str)
    fluxUnits1 = Field(doc="Units for col1", dtype=str, default="nanojansky")
    col2 = Field(doc="Column to subtract", dtype=str)
    fluxUnits2 = Field(doc="Units for col2", dtype=str, default="nanojansky")
    returnMillimags = Field(doc="Use millimags or not?", dtype=bool, default=True)

    @property
    def columns(self):
        return (self.col1, self.col2)

    def __call__(self, df):
        flux1 = df[self.col1].values * u.Unit(self.fluxUnits1)
        mag1 = flux1.to(u.ABmag)

        flux2 = df[self.col2].values * u.Unit(self.fluxUnits2)
        mag2 = flux2.to(u.ABmag)

        magDiff = mag1 - mag2

        if self.returnMillimags:
            magDiff = magDiff*1000.0
        return magDiff


class CalcE1(MultiColumnAction):
    """Calculate E1: (ixx - iyy)/(ixx + iyy)
    This is a shape measurement used for doing QA on the ellipticity
    of the sources."""

    colXx = Field(doc="The column name to get the xx shape component from.",
                  dtype=str,
                  default="ixx")

    colYy = Field(doc="The column name to get the yy shape component from.",
                  dtype=str,
                  default="iyy")

    @property
    def columns(self):
        return (self.colXx, self.colYy)

    def __call__(self, df):
        e1 = (df[self.colXx] - df[self.colYy])/(df[self.colXx] + df[self.colYy])

        return e1


class CalcE2(MultiColumnAction):
    """Calculate E2: 2ixy/(ixx+iyy)
    This is a shape measurement used for doing QA on the ellipticity
    of the sources."""

    colXx = Field(doc="The column name to get the xx shape component from.",
                  dtype=str,
                  default="ixx")

    colYy = Field(doc="The column name to get the yy shape component from.",
                  dtype=str,
                  default="iyy")

    colXy = Field(doc="The column name to get the xy shape component from.",
                  dtype=str,
                  default="ixy")

    @property
    def columns(self):
        return (self.colXx, self.colYy, self.colXy)

    def __call__(self, df):
        e2 = 2*df[self.colXy]/(df[self.colXx] + df[self.colYy])
        return e2


class CalcShapeSize(MultiColumnAction):
    """Calculate a size: (ixx*iyy - ixy**2)**0.25
    This is a size measurement used for doing QA on the ellipticity
    of the sources."""

    colXx = Field(doc="The column name to get the xx shape component from.",
                  dtype=str,
                  default="ixx")

    colYy = Field(doc="The column name to get the yy shape component from.",
                  dtype=str,
                  default="iyy")

    colXy = Field(doc="The column name to get the xy shape component from.",
                  dtype=str,
                  default="ixy")

    @property
    def columns(self):
        return (self.colXx, self.colYy, self.colXy)

    def __call__(self, df):
        size = np.power(df[self.colXx]*df[self.colYy] - df[self.colXy]**2, 0.25)
        return size


class ColorDiff(MultiColumnAction):
    """Calculate the difference between two colors;
    each color is derived from two flux columns.

    The color difference is computed as (color1 - color2) with:

    color1 = color1_mag1 - color1_mag2
    color2 = color2_mag1 - color2_mag2

    where color1_mag1 is the magnitude associated with color1_flux1, etc.

    Parameters
    ----------
    df : `pandas.core.frame.DataFrame`
        The catalog to calculate the color difference from.

    Returns
    -------
    The color difference in millimags.

    Notes
    -----
    The flux columns need to be in units that can be converted
    to janskies. This action doesn't have any calibration
    information and assumes that the fluxes are already
    calibrated.
    """
    color1_flux1 = Field(doc="Column for flux1 to determine color1",
                         dtype=str)
    color1_flux1_units = Field(doc="Units for color1_flux1",
                               dtype=str,
                               default="nanojansky")
    color1_flux2 = Field(doc="Column for flux2 to determine color1",
                         dtype=str)
    color1_flux2_units = Field(doc="Units for color1_flux2",
                               dtype=str,
                               default="nanojansky")
    color2_flux1 = Field(doc="Column for flux1 to determine color2",
                         dtype=str)
    color2_flux1_units = Field(doc="Units for color2_flux1",
                               dtype=str,
                               default="nanojansky")
    color2_flux2 = Field(doc="Column for flux2 to determine color2",
                         dtype=str)
    color2_flux2_units = Field(doc="Units for color2_flux2",
                               dtype=str,
                               default="nanojansky")
    return_millimags = Field(doc="Use millimags or not?",
                             dtype=bool,
                             default=True)

    @property
    def columns(self):
        return (self.color1_flux1,
                self.color1_flux2,
                self.color2_flux1,
                self.color2_flux2)

    def __call__(self, df):
        color1_flux1 = df[self.color1_flux1].values*u.Unit(self.color1_flux1_units)
        color1_mag1 = color1_flux1.to(u.ABmag)

        color1_flux2 = df[self.color1_flux1].values*u.Unit(self.color1_flux2_units)
        color1_mag2 = color1_flux2.to(u.ABmag)

        color2_flux1 = df[self.color2_flux1].values*u.Unit(self.color2_flux1_units)
        color2_mag1 = color2_flux1.to(u.ABmag)

        color2_flux2 = df[self.color2_flux2].values*u.Unit(self.color2_flux2_units)
        color2_mag2 = color2_flux2.to(u.ABmag)

        color1 = color1_mag1 - color1_mag2
        color2 = color2_mag1 - color2_mag2

        color_diff = color1 - color2

        if self.return_millimags:
            color_diff *= 1000.0

        return color_diff


class ColorDiffPull(ColorDiff):
    """Calculate the difference between two colors, scaled by the color error;
    Each color is derived from two flux columns.

    The color difference is computed as (color1 - color2) with:

    color1 = color1_mag1 - color1_mag2
    color2 = color2_mag1 - color2_mag2

    where color1_mag1 is the magnitude associated with color1_flux1, etc.

    The color difference (color1 - color2) is then scaled by the error on
    the color as computed from color1_flux1_err, color1_flux2_err,
    color2_flux1_err, color2_flux2_err.  The errors on color2 may be omitted
    if the comparison is between an "observed" catalog and a "truth" catalog.

    Parameters
    ----------
    df : `pandas.core.frame.DataFrame`
        The catalog to calculate the color difference from.

    Returns
    -------
    The color difference scaled by the error.

    Notes
    -----
    The flux columns need to be in units that can be converted
    to janskies. This action doesn't have any calibration
    information and assumes that the fluxes are already
    calibrated.
    """
    color1_flux1_err = Field(doc="Error column for flux1 for color1",
                             dtype=str)
    color1_flux2_err = Field(doc="Error column for flux2 for color1",
                             dtype=str)
    color2_flux1_err = Field(doc="Error column for flux1 for color2",
                             dtype=str,
                             default="")
    color2_flux2_err = Field(doc="Error column for flux2 for color2",
                             dtype=str,
                             default="")

    @property
    def columns(self):
        cols = (self.color1_flux1,
                self.color1_flux2,
                self.color2_flux1,
                self.color2_flux2,
                self.color1_flux1_err,
                self.color1_flux2_err)

        if self.color2_flux1_err and self.color2_flux2_err:
            cols = cols + (self.color2_flux1_err,
                           self.color2_flux2_err)

        return cols

    def __call__(self, df):
        k = 2.5/np.log(10.)

        color1_flux1 = df[self.color1_flux1].values*u.Unit(self.color1_flux1_units)
        color1_mag1 = color1_flux1.to(u.ABmag)
        color1_mag1_err = k*df[self.color1_flux1_err].values/df[self.color1_flux1].values

        color1_flux2 = df[self.color1_flux1].values*u.Unit(self.color1_flux2_units)
        color1_mag2 = color1_flux2.to(u.ABmag)
        color1_mag2_err = k*df[self.color1_flux2_err].values/df[self.color1_flux2].values

        color2_flux1 = df[self.color2_flux1].values*u.Unit(self.color2_flux1_units)
        color2_mag1 = color2_flux1.to(u.ABmag)
        if self.color2_flux1_err:
            color2_mag1_err = k*df[self.color2_flux1_err].values/df[self.color2_flux1].values
        else:
            color2_mag1_err = 0.0

        color2_flux2 = df[self.color2_flux2].values*u.Unit(self.color2_flux2_units)
        color2_mag2 = color2_flux2.to(u.ABmag)
        if self.color2_flux2_err:
            color2_mag2_err = k*df[self.color2_flux2_err].values/df[self.color2_flux2].values
        else:
            color2_mag2_err = 0.0

        color1 = color1_mag1 - color1_mag2
        err1_sq = color1_mag1_err**2. + color1_mag2_err**2.
        color2 = color2_mag1 - color2_mag2
        err2_sq = color2_mag1_err**2. + color2_mag2_err**2.

        color_diff = color1 - color2

        pull = color_diff/np.sqrt(err1_sq + err2_sq)

        return pull
