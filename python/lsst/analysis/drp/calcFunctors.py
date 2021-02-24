from lsst.pex.config import Field
from lsst.pipe.tasks.dataFrameActions import DivideColumns, SingleColumnAction


class SNCalculator(SingleColumnAction):
    # Not needed yet but added for the change of catalogues
    band = Field(doc="Which band to use.",
                 dtype=str,
                 default="i")

    def __call__(self, df, **kwargs):
        return df[self.column] / df[f"{self.column}_instfluxErr"]

    def setDefaults(self):
        super().setDefaults()
        self.column = "base_PsfFlux_instFlux"


class KronFluxDivPsfFlux(DivideColumns):
    """Divide the Kron instFLux by the PSF instFlux"""

    def __call__(self, cat, **kwargs):
        return self.actions.numerator(cat, kwargs) / self.actions.divisor(cat, kwargs)

    def setDefaults(self):
        super().setDefaults()
        self.actions.ColA.column = "ext_photometryKron_KronFlux_instFlux"
        self.actions.ColB.column = "base_PsfFlux_instFlux"
