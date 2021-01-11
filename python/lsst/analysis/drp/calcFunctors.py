__all__ = ["SNCalculator", "KronFluxDivPsfFlux"]

from lsst.pipe.tasks.dataFrameActions import DivideColumns


class SNCalculator(DivideColumns):
    """Calculate the signal to noise by default the i band PSF flux is used"""

    def setDefaults(self):
        super().setDefaults()
        self.actions.ColA.column = "iPsFlux"
        self.actions.ColB.column = "iPsFluxErr"


class KronFluxDivPsfFlux(DivideColumns):
    """Divide the Kron instFlux by the PSF instFlux"""

    def setDefaults(self):
        super().setDefaults()
        self.actions.ColA.column = "iKronFlux"
        self.actions.ColB.column = "iPsFlux"
