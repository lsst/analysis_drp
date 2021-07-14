__all__ = ["SNCalculator", "KronFluxDivPsfFlux", "MagDiff", "PsfTraceSizeDiff"]

from lsst.pipe.tasks.dataFrameActions import DivideColumns, MultiColumnAction
from lsst.pex.config import Field
import numpy as np


class SNCalculator(DivideColumns):
    """Calculate the signal to noise by default the i band PSF flux is used"""

    def setDefaults(self):
        super().setDefaults()
        self.colA.column = "iPsFlux"
        self.colB.column = "iPsFluxErr"


class KronFluxDivPsfFlux(DivideColumns):
    """Divide the Kron instFlux by the PSF instFlux"""

    def setDefaults(self):
        super().setDefaults()
        self.colA.column = "iKronFlux"
        self.colB.column = "iPsFlux"


class MagDiff(MultiColumnAction):
    """Calculate the difference between two magntiudes"""

    col1 = Field(doc="Column to subtract from", dtype=str)
    col2 = Field(doc="Column to subtract", dtype=str)

    @property
    def columns(self):
        return (self.col1, self.col2)

    def __call__(self, df):
        mag1 = -2.5*np.log10((df[self.col1]*1e-9) / 3631.0)
        mag2 = -2.5*np.log10((df[self.col2]*1e-9) / 3631.0)

        return (mag1 - mag2)*1000.0


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
