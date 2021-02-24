__all__ = ("HighSnSelector", "LowSnSelector")

from lsst.pex.config import ListField, Field
from lsst.pipe.tasks.dataFrameActions import DataFrameAction


class MainFlagSelector(DataFrameAction):
    """The main set of flags to use to select valid sources for QA"""
    filterColumns = ListField(doc="Names of the flag columns to filter on", dtype=str, optional=False,
                              default=["base_PixelFlags_flag", "base_PsfFlux_flag"])

    @property
    def columns(self):
        yield from self.filterColumns

    def __call__(self, df, **kwargs):
        """Select on the given flags"""
        result = None
        for flag in self.filterColumns:
            selected = df[flag]
            if result is None:
                result = selected
            else:
                result &= selected
        return result


class BaseSNRSelector(DataFrameAction):
    fluxField = Field(doc="Flux field to use in SNR calculation", dtype=str,
                      default="base_PsfFlux_instFlux", optional=False)
    errField = Field(doc="Flux err field to use in SNR calculation", dtype=str,
                     default="base_PsfFlux_instFluxErr", optional=False)
    threshold = Field(doc="The signal to noise threshold to select sources",
                      dtype=float,
                      optional=False)

    @property
    def columns(self):
        return (self.fluxField, self.errField)


class HighSnSelector(BaseSNRSelector):
    """Select high signal to noise sources, in PSF flux"""

    def __call__(self, df, **kwargs):
        """Selects sources with PSF SN ratio above self.threshold"""

        return (df[self.fluxField] / df[self.errField]) > self.threshold

    def setDefaults(self):
        super().setDefaults()
        self.threshold = 2700


class LowSnSelector(BaseSNRSelector):
    """Select lower signal to noise sources, in PSF flux"""

    def __call__(self, df, **kwargs):
        """Selects sources with PSF SN ratio below self.threshold"""

        return (df[self.fluxField] / df[self.errField]) > self.threshold

    def setDefaults(self):
        super().setDefaults()
        self.threshold = 500
