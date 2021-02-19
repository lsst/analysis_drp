from lsst.analysis.drp.configStructField import Config, Field


class MainFlagSelector(Config):
    """The main set of flags to use to select valid sources for QA"""

    def select(self, cat):
        """Select on the given flags"""

        keep = ((cat["base_PixelFlags_flag"] == 0) & (cat["base_PsfFlux_flag"] == 0))
        return keep


class HighSnSelector(Config):
    """Select high signal to noise sources, in PSF flux"""

    threshold = Field(doc="The signal to noise threshold to select sources above.",
                      dtype=float,
                      default=2700)

    def select(self, cat):
        """Selects sources with PSF SN ratio above self.threshold"""

        use = (cat["base_PsfFlux_instFlux"] / cat["base_PsfFlux_instFluxErr"] > self.threshold)
        return use


class LowSnSelector(Config):
    """Select lower signal to noise sources, in PSF flux"""

    threshold = Field(doc="The signal to noise threshold to select sources above.",
                      dtype=float,
                      default=500)

    def select(self, cat):
        """Selects sources with PSF SN ratio above self.threshold"""

        use = (cat["base_PsfFlux_instFlux"] / cat["base_PsfFlux_instFluxErr"] > self.threshold)
        return use
