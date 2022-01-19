__all__ = ["SNCalculator", "KronFluxDivPsfFlux", "MagDiff"]

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
