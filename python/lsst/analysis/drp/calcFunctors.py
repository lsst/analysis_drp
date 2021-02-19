from lsst.analysis.drp.configStructField import Config, Field
import numpy as np


class SNCalculator(Config):

    # Not needed yet but added for the change of catalogues
    band = Field(doc="Which band to use.",
                 dtype=str,
                 default="i")

    def calculate(self, cat):
        """Calculate the signal to noise ratio for the PSF flux"""

        sn = cat["base_PsfFlux_instFlux"] / cat["base_PsfFlux_instFluxErr"]
        return sn, "PsSn"


class RadToDeg(Config):
    """Convert a column in radians to degrees"""

    colName = Field(doc="Name of the column to convert from radians to degrees.",
                    dtype=str,
                    default="coord_ra")

    def setDefaults(self):
        super().setDefaults()

    def calculate(self, cat):
        """Convert radians to degrees and return the column name
           with '_deg' appended"""
        return np.rad2deg(cat[self.colName]), self.colName + "_deg"


class KronFluxDivPsfFlux(Config):
    """Divide the Kron instFLux by the PSF instFlux"""

    def calculate(self, cat):
        return cat["ext_photometryKron_KronFlux_instFlux"] / cat["base_PsfFlux_instFlux"]
