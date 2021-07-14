__all__ = ["SNCalculator", "KronFluxDivPsfFlux", "MagDiff", "PsfTraceSizeDiff"]

from lsst.pipe.tasks.dataFrameActions import DivideColumns, MultiColumnAction
from lsst.pex.config import Field
from astropy import units as u
import numpy as np


class SNCalculator(DivideColumns):
    """Calculate the signal to noise by default the i band PSF flux is used"""

    def setDefaults(self):
        super().setDefaults()
        self.colA.column = "i_PsfFlux"
        self.colB.column = "i_PsfFluxErr"


class KronFluxDivPsfFlux(DivideColumns):
    """Divide the Kron instFlux by the PSF instFlux"""

    def setDefaults(self):
        super().setDefaults()
        self.colA.column = "i_KronFlux"
        self.colB.column = "i_PsfFlux"


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


class PsfTraceSizeDiff(MultiColumnAction):
    """Calculate the trace radius size difference (%) between object and
    PSF model.
    """

    col1 = Field(doc="Object size xx", dtype=str)
    col2 = Field(doc="Object size yy", dtype=str)

    @property
    def columns(self):
        return (self.col1, self.col2, self.col1 + "Psf", self.col2 + "Psf")

    def __call__(self, df):
        srcSize = np.sqrt(0.5*(df[self.col1] + df[self.col2]))
        psfSize = np.sqrt(0.5*(df[self.col1 + "Psf"] + df[self.col2 + "Psf"]))
        sizeDiff = 100*(srcSize - psfSize)/(0.5*(srcSize + psfSize))
        return np.array(sizeDiff)

class DistCalc(MultiColumnAction):
    """
    Compute distance scales
    """

    col1 =
    col2
    col3
    col4
