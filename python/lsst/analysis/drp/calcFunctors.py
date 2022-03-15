# This file is part of analysis_drp.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["SNCalculator", "KronFluxDivPsfFlux", "MagDiff", "ColorDiff", "ColorDiffPull",
           "ExtinctionCorrectedMagDiff", "CalcE", "CalcE1", "CalcE2", "CalcEDiff", "CalcShapeSize",
           "CalcRhoStatistics", ]

import logging

import numpy as np
import treecorr
from astropy import units as u

from lsst.pex.config import ChoiceField, ConfigField, DictField, Field, FieldValidationError
from lsst.pipe.tasks.configurableActions import ConfigurableActionField
from lsst.pipe.tasks.dataFrameActions import (CoordColumn, DataFrameAction, DivideColumns,
                                              FractionalDifferenceColumns, MultiColumnAction,
                                              SingleColumnAction,)

from ._treecorrConfig import BinnedCorr2Config

_LOG = logging.getLogger(__name__)


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
            magDiff = magDiff.to(u.mmag)

        return magDiff


class ExtinctionCorrectedMagDiff(DataFrameAction):
    """Compute the difference between two magnitudes and correct for extinction

    By default bands are derived from the <band>_ prefix on flux columns,
    per the naming convention in the Object Table:
    e.g. the band of 'g_psfFlux' is 'g'. If column names follow another
    convention, bands can alternatively be supplied via the band1 or band2
    config parameters.
    If band1 and band2 are supplied, the flux column names are ignored.
    """

    magDiff = ConfigurableActionField(doc="Action that returns a difference in magnitudes",
                                          default=MagDiff, dtype=DataFrameAction)
    ebvCol = Field(doc="E(B-V) Column Name", dtype=str, default="ebv")
    band1 = Field(doc="Optional band for magDiff.col1. Supercedes column name prefix",
                  dtype=str, optional=True, default=None)
    band2 = Field(doc="Optional band for magDiff.col2. Supercedes column name prefix",
                  dtype=str, optional=True, default=None)
    extinctionCoeffs = DictField(
        doc="Dictionary of extinction coefficients for conversion from E(B-V) to extinction, A_band."
            "Key must be the band",
        keytype=str, itemtype=float, optional=True,
        default=None)

    @property
    def columns(self):
        return self.magDiff.columns + (self.ebvCol,)

    def __call__(self, df):
        diff = self.magDiff(df)
        if not self.extinctionCoeffs:
            _LOG.warning("No extinction Coefficients. Not applying extinction correction")
            return diff

        col1Band = self.band1 if self.band1 else self.magDiff.col1.split('_')[0]
        col2Band = self.band2 if self.band2 else self.magDiff.col2.split('_')[0]

        for band in (col1Band, col1Band):
            if band not in self.extinctionCoeffs:
                _LOG.warning("%s band not found in coefficients dictionary: %s"
                             " Not applying extinction correction", band, self.extinctionCoeffs)
                return diff

        av1 = self.extinctionCoeffs[col1Band]
        av2 = self.extinctionCoeffs[col2Band]

        ebv = df[self.ebvCol].values
        correction = (av1 - av2) * ebv * u.mag

        if self.magDiff.returnMillimags:
            correction = correction.to(u.mmag)

        return diff - correction


class CalcE(MultiColumnAction):
    """Calculate a complex value representation of the ellipticity.

    The complex ellipticity is typically defined as
    e = |e|exp(j*2*theta) = ((Ixx - Iyy) + j*(2*Ixy))/(Ixx + Iyy), where j is
    the square root of -1 and Ixx, Iyy, Ixy are second-order central moments.
    This is sometimes referred to as distortion, and denoted by e = (e1, e2)
    in GalSim and referred to as chi-type ellipticity following the notation
    in Eq. 4.4 of Bartelmann and Schneider (2001). The other definition differs
    in normalization. It is referred to as shear, and denoted by g = (g1, g2)
    in GalSim and referred to as epsilon-type ellipticity again following the
    notation in Eq. 4.10 of Bartelmann and Schneider (2001). It is defined as
    g = ((Ixx - Iyy) + j*(2*Ixy))/(Ixx + Iyy + 2sqrt(Ixx*Iyy - Ixy**2)).

    The shear measure is unbiased in weak-lensing shear, but may exclude some
    objects in the presence of noisy moment estimates. The distortion measure
    is biased in weak-lensing distortion, but does not suffer from selection
    artifacts.

    References
    ----------
    [1] Bartelmann, M. and Schneider, P., “Weak gravitational lensing”,
    Physics Reports, vol. 340, no. 4–5, pp. 291–472, 2001.
    doi:10.1016/S0370-1573(00)00082-X; https://arxiv.org/abs/astro-ph/9912508

    Notes
    -----

    1. This is a shape measurement used for doing QA on the ellipticity
    of the sources.

    2. For plotting purposes we might want to plot |E|*exp(i*theta).
    If `halvePhaseAngle` config parameter is set to `True`, then
    the returned quantity therefore corresponds to |E|*exp(i*theta).

    See Also
    --------
    CalcE1
    CalcE2
    """

    colXx = Field(
        doc="The column name to get the xx shape component from.",
        dtype=str,
        default="ixx",
    )

    colYy = Field(
        doc="The column name to get the yy shape component from.",
        dtype=str,
        default="iyy",
    )

    colXy = Field(
        doc="The column name to get the xy shape component from.",
        dtype=str,
        default="ixy",
    )

    ellipticityType = ChoiceField(
        doc="The type of ellipticity to calculate",
        dtype=str,
        allowed={"chi": ("Distortion, defined as (Ixx - Iyy + 2j*Ixy)/"
                         "(Ixx + Iyy)"
                         ),
                 "epsilon": ("Shear, defined as (Ixx - Iyy + 2j*Ixy)/"
                             "(Ixx + Iyy + 2*sqrt(Ixx*Iyy - Ixy**2))"
                             ),
                 },
        default="chi",
    )

    halvePhaseAngle = Field(
        doc="Divide the phase angle by 2? Suitable for quiver plots.",
        dtype=bool,
        default=False,
    )

    @property
    def columns(self):
        return (self.colXx, self.colYy, self.colXy)

    def __call__(self, df):
        e = (df[self.colXx] - df[self.colYy]) + 1j*(2*df[self.colXy])
        denom = (df[self.colXx] + df[self.colYy])

        if self.ellipticityType == "epsilon":
            denom += 2*np.sqrt(df[self.colXx]*df[self.colYy] - df[self.colXy]**2)

        e /= denom

        if self.halvePhaseAngle:
            # Ellipiticity is |e|*exp(i*2*theta), but we want to return
            # |e|*exp(i*theta). So we multiply by |e| and take its square root
            # instead of the more expensive trig calls.
            e *= np.abs(e)
            return np.sqrt(e)
        else:
            return e


class CalcEDiff(DataFrameAction):
    """Calculate the difference of two ellipticities as a complex quantity.

    The complex ellipticity difference between e_A and e_B is defined as
    e_A - e_B = de = |de|exp(j*2*theta).

    See Also
    --------
    CalcE

    Notes
    -----

    1. This is a shape measurement used for doing QA on the ellipticity
    of the sources.

    2. For plotting purposes we might want to plot |de|*exp(j*theta).
    If `halvePhaseAngle` config parameter is set to `True`, then
    the returned quantity therefore corresponds to |e|*exp(j*theta).
    """
    colA = ConfigurableActionField(
        doc="Ellipticity to subtract from",
        dtype=MultiColumnAction,
        default=CalcE,
    )

    colB = ConfigurableActionField(
        doc="Ellipticity to subtract",
        dtype=MultiColumnAction,
        default=CalcE,
    )

    halvePhaseAngle = Field(
        doc="Divide the phase angle by 2? Suitable for quiver plots.",
        dtype=bool,
        default=False,
    )

    @property
    def columns(self):
        yield from self.colA.columns
        yield from self.colB.columns

    def validate(self):
        super().validate()
        if self.colA.ellipticityType != self.colB.ellipticityType:
            msg = "Both the ellipticities in CalcEDiff must have the same type."
            raise FieldValidationError(self.colB.__class__.ellipticityType, self, msg)

    def __call__(self, df):
        eMeas = self.colA(df)
        ePSF = self.colB(df)
        eDiff = eMeas - ePSF
        if self.halvePhaseAngle:
            # Ellipiticity is |e|*exp(i*2*theta), but we want to return
            # |e|*exp(j*theta). So we multiply by |e| and take its square root
            # instead of the more expensive trig calls.
            eDiff *= np.abs(eDiff)
            return np.sqrt(eDiff)
        else:
            return eDiff


class CalcE1(MultiColumnAction):
    """Calculate chi-type e1 = (Ixx - Iyy)/(Ixx + Iyy) or
    epsilon-type g1 = (Ixx - Iyy)/(Ixx + Iyy + 2sqrt(Ixx*Iyy - Ixy**2)).

    See Also
    --------
    CalcE
    CalcE2

    Note
    ----
    This is a shape measurement used for doing QA on the ellipticity
    of the sources.
    """

    colXx = Field(
        doc="The column name to get the xx shape component from.",
        dtype=str,
        default="ixx",
    )

    colYy = Field(
        doc="The column name to get the yy shape component from.",
        dtype=str,
        default="iyy",
    )

    colXy = Field(
        doc="The column name to get the xy shape component from.",
        dtype=str,
        default="ixy",
        optional=True,
    )

    ellipticityType = ChoiceField(
        doc="The type of ellipticity to calculate",
        dtype=str,
        allowed={"chi": "Distortion, defined as (Ixx - Iyy)/(Ixx + Iyy)",
                 "epsilon": ("Shear, defined as (Ixx - Iyy)/"
                             "(Ixx + Iyy + 2*sqrt(Ixx*Iyy - Ixy**2))"
                             ),
                 },
        default="chi",
    )

    @property
    def columns(self):
        if self.ellipticityType == "chi":
            return (self.colXx, self.colYy)
        else:
            return (self.colXx, self.colYy, self.colXy)

    def __call__(self, df):
        denom = df[self.colXx] + df[self.colYy]
        if self.ellipticityType == "epsilon":
            denom += 2*np.sqrt(df[self.colXx] * df[self.colYy] - df[self.colXy]**2)
        e1 = (df[self.colXx] - df[self.colYy])/denom

        return e1

    def validate(self):
        super().validate()
        if self.ellipticityType == "epsilon" and self.colXy is None:
            msg = "colXy is required for epsilon-type shear ellipticity"
            raise FieldValidationError(self.__class__.colXy, self, msg)


class CalcE2(MultiColumnAction):
    """Calculate chi-type e2 = 2Ixy/(Ixx+Iyy) or
    epsilon-type g2 = 2Ixy/(Ixx+Iyy+2sqrt(Ixx*Iyy - Ixy**2)).

    See Also
    --------
    CalcE
    CalcE1

    Note
    ----
    This is a shape measurement used for doing QA on the ellipticity
    of the sources.
    """

    colXx = Field(
        doc="The column name to get the xx shape component from.",
        dtype=str,
        default="ixx",
    )

    colYy = Field(
        doc="The column name to get the yy shape component from.",
        dtype=str,
        default="iyy",
    )

    colXy = Field(
        doc="The column name to get the xy shape component from.",
        dtype=str,
        default="ixy",
    )

    ellipticityType = ChoiceField(
        doc="The type of ellipticity to calculate",
        dtype=str,
        allowed={"chi": "Distortion, defined as 2*Ixy/(Ixx + Iyy)",
                 "epsilon": ("Shear, defined as 2*Ixy/"
                             "(Ixx + Iyy + 2*sqrt(Ixx*Iyy - Ixy**2))"
                             ),
                 },
        default="chi",
    )

    @property
    def columns(self):
        return (self.colXx, self.colYy, self.colXy)

    def __call__(self, df):
        denom = df[self.colXx] + df[self.colYy]
        if self.ellipticityType == "epsilon":
            denom += 2*np.sqrt(df[self.colXx] * df[self.colYy] - df[self.colXy]**2)
        e2 = 2*df[self.colXy]/denom
        return e2


class CalcShapeSize(MultiColumnAction):
    """Calculate a size: (Ixx*Iyy - Ixy**2)**0.25 OR (0.5*(Ixx + Iyy))**0.5

    The square of size measure is typically expressed either as the arithmetic
    mean of the eigenvalues of the moment matrix (trace radius) or as the
    geometric mean of the eigenvalues (determinant radius), which can be
    specified using the ``sizeType`` parameter. Both of these measures give the
    `sigma^2` parameter for a 2D Gaussian.

    Since lensing preserves surface brightness, the determinant radius relates
    the magnification cleanly as it is derived from the area of isophotes, but
    have a slightly higher chance of being NaNs for noisy moment estimates.

    Note
    ----
    This is a size measurement used for doing QA on the ellipticity
    of the sources.
    """

    colXx = Field(
        doc="The column name to get the xx shape component from.",
        dtype=str,
        default="ixx",
    )

    colYy = Field(
        doc="The column name to get the yy shape component from.",
        dtype=str,
        default="iyy",
    )

    colXy = Field(
        doc="The column name to get the xy shape component from.",
        dtype=str,
        default="ixy",
        optional=True,
    )

    sizeType = ChoiceField(
        doc="The type of size to calculate",
        dtype=str,
        default="determinant",
        allowed={"trace": "trace radius",
                 "determinant": "determinant radius",
                 },
    )

    @property
    def columns(self):
        if self.sizeType == "trace":
            return (self.colXx, self.colYy,)
        else:
            return (self.colXx, self.colYy, self.colXy)

    def __call__(self, df):
        if self.sizeType == "trace":
            size = np.power(0.5*(df[self.colXx] + df[self.colYy]), 0.5)
        else:
            size = np.power(df[self.colXx]*df[self.colYy] - df[self.colXy]**2, 0.25)

        return size

    def validate(self):
        super().validate()
        if self.sizeType == "determinant" and self.colXy is None:
            msg = "colXy is required for determinant-type size"
            raise FieldValidationError(self.__class__.colXy, self, msg)


class CalcRhoStatistics(DataFrameAction):
    r"""Calculate Rho statistics.

    Rho statistics refer to a collection of correlation functions involving
    PSF ellipticity and size residuals. They quantify the contribution from PSF
    leakage due to errors in PSF modeling to the weak lensing shear correlation
    functions. The standard rho statistics are indexed from 1 to 5, and
    this action calculates a sixth rho statistic, indexed 0.

    Notes
    -----
    The exact definitions of rho statistics as defined in [1]_ are given below.
    In addition to these five, we also compute the auto-correlation function of
    the fractional size residuals and call it as the :math:`\rho_0( \theta )`.

    .. math::

       \rho_0(\theta) &= \left\langle \frac{\delta T_{PSF}}{T_{PSF}}(x) \frac{\delta T_{PSF}}{T_{PSF}}(x+\theta) \right\rangle  # noqa: E501, W505

       \rho_1(\theta) &= \langle \delta e^*_{PSF}(x) \delta e_{PSF}(x+\theta) \rangle  # noqa: W505

       \rho_2(\theta) &= \langle e^*_{PSF}(x) \delta e_{PSF}(x+\theta) \rangle

       \rho_3(\theta) &= \left\langle (e^*_{PSF}\frac{\delta T_{PSF}}{T_{PSF}}(x)) \delta e_{PSF}(x+\theta) \right\rangle  # noqa: E501, W505

       \rho_4(\theta) &= \left\langle (\delta e^*_{PSF}(x) (e_{PSF}\frac{\delta T_{PSF}}{T_{PSF}}(x+\theta)) \right\rangle  # noqa: E501, W505

       \rho_5(\theta) &= \left\langle (e^*_{PSF}(x) (e_{PSF}\frac{\delta T_{PSF}}{T_{PSF}}(x+\theta)) \right\rangle  # noqa: E501, W505

    The definition of ellipticity used in [1]_ correspond to ``epsilon``-type ellipticity, which is typically
    smaller by a factor of 4 than using ``chi``-type ellipticity.

    References
    ----------
    .. [1] Jarvis, M., Sheldon, E., Zuntz, J., Kacprzak, T., Bridle, S. L., et. al (2016).  # noqa: W501
           The DES Science Verification weak lensing shear catalogues.
           MNRAS, 460, 2245–2281.
           https://doi.org/10.1093/mnras/stw990;
           https://arxiv.org/abs/1507.05603
    """

    colRa = ConfigurableActionField(doc="RA column", dtype=SingleColumnAction, default=CoordColumn)

    colDec = ConfigurableActionField(doc="Dec column", dtype=SingleColumnAction, default=CoordColumn)

    colXx = Field(
        doc="The column name to get the xx shape component from.",
        dtype=str,
        default="ixx"
    )

    colYy = Field(
        doc="The column name to get the yy shape component from.",
        dtype=str,
        default="iyy"
    )

    colXy = Field(
        doc="The column name to get the xy shape component from.",
        dtype=str,
        default="ixy"
    )

    colPsfXx = Field(
        doc="The column name to get the PSF xx shape component from.",
        dtype=str,
        default="ixxPSF"
    )

    colPsfYy = Field(
        doc="The column name to get the PSF yy shape component from.",
        dtype=str,
        default="iyyPSF"
    )

    colPsfXy = Field(
        doc="The column name to get the PSF xy shape component from.",
        dtype=str,
        default="ixyPSF"
    )

    ellipticityType = ChoiceField(
        doc="The type of ellipticity to calculate",
        dtype=str,
        allowed={"chi": "Distortion, defined as (Ixx - Iyy)/(Ixx + Iyy)",
                 "epsilon": ("Shear, defined as (Ixx - Iyy)/"
                             "(Ixx + Iyy + 2*sqrt(Ixx*Iyy - Ixy**2))"
                             ),
                 },
        default="chi",
    )

    sizeType = ChoiceField(
        doc="The type of size to calculate",
        dtype=str,
        default="trace",
        allowed={"trace": "trace radius",
                 "determinant": "determinant radius",
                 },
    )

    treecorr = ConfigField(
        doc="TreeCorr configuration",
        dtype=BinnedCorr2Config,
    )

    def setDefaults(self):
        super().setDefaults()
        self.treecorr = BinnedCorr2Config()
        self.treecorr.sep_units = "arcmin"
        self.treecorr.metric = "Arc"
        # Note: self.treecorr is not configured completely at this point.
        # Exactly three of nbins, bin_size, min_sep, max_sep need to be set.
        # These are expected to be set in the tasks that use this action.

    @property
    def columns(self):
        return (
            self.colXx,
            self.colYy,
            self.colXy,
            self.colPsfXx,
            self.colPsfYy,
            self.colPsfXy,
            self.colRa.column,
            self.colDec.column,
        )

    def __call__(self, df):
        # Create instances of various actions.
        calcEMeas = CalcE(
            colXx=self.colXx,
            colYy=self.colYy,
            colXy=self.colXy,
            ellipticityType=self.ellipticityType,
        )
        calcEpsf = CalcE(
            colXx=self.colPsfXx,
            colYy=self.colPsfYy,
            colXy=self.colPsfXy,
            ellipticityType=self.ellipticityType,
        )

        calcEDiff = CalcEDiff(colA=calcEMeas, colB=calcEpsf)

        calcSizeResiduals = FractionalDifferenceColumns(
            colA=CalcShapeSize(
                colXx=self.colXx,
                colYy=self.colYy,
                colXy=self.colXy,
                sizeType=self.sizeType,
            ),
            colB=CalcShapeSize(
                colXx=self.colPsfXx,
                colYy=self.colPsfYy,
                colXy=self.colPsfXy,
                sizeType=self.sizeType,
            ),
        )

        # Call the actions on the dataframe.
        eMEAS = calcEMeas(df)
        e1, e2 = np.real(eMEAS), np.imag(eMEAS)
        eRes = calcEDiff(df)
        e1Res, e2Res = np.real(eRes), np.imag(eRes)
        sizeRes = calcSizeResiduals(df)

        # Scale the sizeRes by ellipticities
        e1SizeRes = e1*sizeRes
        e2SizeRes = e2*sizeRes

        # Package the arguments to capture auto-/cross-correlations for the
        # Rho statistics.
        args = {
            0: (sizeRes, None),
            1: (e1Res, e2Res, None, None),
            2: (e1, e2, e1Res, e2Res),
            3: (e1SizeRes, e2SizeRes, None, None),
            4: (e1Res, e2Res, e1SizeRes, e2SizeRes),
            5: (e1, e2, e1SizeRes, e2SizeRes),
        }

        ra = self.colRa(df)
        dec = self.colDec(df)

        # If RA and DEC are not in radians, they are assumed to be in degrees.
        if self.colRa.inRadians:
            ra *= 180.0/np.pi
        if self.colDec.inRadians:
            dec *= 180.0/np.pi

        # Convert the self.treecorr Config to a kwarg dict.
        treecorrKwargs = self.treecorr.toDict()

        # Pass the appropriate arguments to the correlator and build a dict
        rhoStats = {
            rhoIndex: self._corrSpin2(ra, dec, *(args[rhoIndex]), **treecorrKwargs)
            for rhoIndex in range(1, 6)
        }
        rhoStats[0] = self._corrSpin0(ra, dec, *(args[0]), **treecorrKwargs)

        return rhoStats

    @classmethod
    def _corrSpin0(cls, ra, dec, k1, k2=None, raUnits="degrees", decUnits="degrees", **treecorrKwargs):
        """Function to compute correlations between at most two scalar fields.

        This is used to compute Rho0 statistics, given the appropriate spin-0
        (scalar) fields, usually fractional size residuals.

        Parameters
        ----------
        ra : `numpy.array`
            The right ascension values of entries in the catalog.
        dec : `numpy.array`
            The declination values of entries in the catalog.
        k1 : `numpy.array`
            The primary scalar field.
        k2 : `numpy.array`, optional
            The secondary scalar field.
            Autocorrelation of the primary field is computed if `None`.
        raUnits : `str`, optional
            Unit of the right ascension values. Valid options are
            "degrees", "arcmin", "arcsec", "hours" or "radians".
        decUnits : `str`, optional
            Unit of the declination values. Valid options are
            "degrees", "arcmin", "arcsec", "hours" or "radians".
        **treecorrKwargs
            Keyword arguments to be passed to `treecorr.KKCorrelation`.

        Returns
        -------
        xy : `treecorr.KKCorrelation`
            A `treecorr.KKCorrelation` object containing the correlation
            function.
        """

        xy = treecorr.KKCorrelation(**treecorrKwargs)
        catA = treecorr.Catalog(ra=ra, dec=dec, k=k1, ra_units=raUnits,
                                dec_units=decUnits)
        if k2 is None:
            # Calculate the auto-correlation
            xy.process(catA)
        else:
            catB = treecorr.Catalog(ra=ra, dec=dec, k=k2, ra_units=raUnits,
                                    dec_units=decUnits)
            # Calculate the cross-correlation
            xy.process(catA, catB)

        return xy

    @classmethod
    def _corrSpin2(cls, ra, dec, g1a, g2a, g1b=None, g2b=None,
                   raUnits="degrees", decUnits="degrees", **treecorrKwargs):
        """Function to compute correlations between shear-like fields.

        This is used to compute Rho statistics, given the appropriate spin-2
        (shear-like) fields.

        Parameters
        ----------
        ra : `numpy.array`
            The right ascension values of entries in the catalog.
        dec : `numpy.array`
            The declination values of entries in the catalog.
        g1a : `numpy.array`
            The first component of the primary shear-like field.
        g2a : `numpy.array`
            The second component of the primary shear-like field.
        g1b : `numpy.array`, optional
            The first component of the secondary shear-like field.
            Autocorrelation of the primary field is computed if `None`.
        g2b : `numpy.array`, optional
            The second component of the secondary shear-like field.
            Autocorrelation of the primary field is computed if `None`.
        raUnits : `str`, optional
            Unit of the right ascension values. Valid options are
            "degrees", "arcmin", "arcsec", "hours" or "radians".
        decUnits : `str`, optional
            Unit of the declination values. Valid options are
            "degrees", "arcmin", "arcsec", "hours" or "radians".
        **treecorrKwargs
            Keyword arguments to be passed to `treecorr.GGCorrelation`.

        Returns
        -------
        xy : `treecorr.GGCorrelation`
            A `treecorr.GGCorrelation` object containing the correlation
            function.
        """
        xy = treecorr.GGCorrelation(**treecorrKwargs)
        catA = treecorr.Catalog(ra=ra, dec=dec, g1=g1a, g2=g2a, ra_units=raUnits, dec_units=decUnits)
        if g1b is None or g2b is None:
            # Calculate the auto-correlation
            xy.process(catA)
        else:
            catB = treecorr.Catalog(ra=ra, dec=dec, g1=g1b, g2=g2b, ra_units=raUnits, dec_units=decUnits)
            # Calculate the cross-correlation
            xy.process(catA, catB)

        return xy


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
        color1_mag1 = color1_flux1.to(u.ABmag).value

        color1_flux2 = df[self.color1_flux2].values*u.Unit(self.color1_flux2_units)
        color1_mag2 = color1_flux2.to(u.ABmag).value

        color2_flux1 = df[self.color2_flux1].values*u.Unit(self.color2_flux1_units)
        color2_mag1 = color2_flux1.to(u.ABmag).value

        color2_flux2 = df[self.color2_flux2].values*u.Unit(self.color2_flux2_units)
        color2_mag2 = color2_flux2.to(u.ABmag).value

        color1 = color1_mag1 - color1_mag2
        color2 = color2_mag1 - color2_mag2

        color_diff = color1 - color2

        if self.return_millimags:
            color_diff = color_diff*1000

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
                             dtype=str,
                             default="")
    color1_flux2_err = Field(doc="Error column for flux2 for color1",
                             dtype=str,
                             default="")
    color2_flux1_err = Field(doc="Error column for flux1 for color2",
                             dtype=str,
                             default="")
    color2_flux2_err = Field(doc="Error column for flux2 for color2",
                             dtype=str,
                             default="")

    def validate(self):
        super().validate()

        color1_errors = False
        color2_errors = False

        if self.color1_flux1_err and self.color1_flux2_err:
            color1_errors = True
        elif ((self.color1_flux1_err and not self.color1_flux2_err)
              or (not self.color1_flux1_err and self.color1_flux2_err)):
            msg = "Must set both color1_flux1_err and color1_flux2_err if either is set."
            raise FieldValidationError(self.__class__.color1_flux1_err, self, msg)
        if self.color2_flux1_err and self.color2_flux2_err:
            color2_errors = True
        elif ((self.color2_flux1_err and not self.color2_flux2_err)
              or (not self.color2_flux1_err and self.color2_flux2_err)):
            msg = "Must set both color2_flux1_err and color2_flux2_err if either is set."
            raise FieldValidationError(self.__class__.color2_flux1_err, self, msg)

        if not color1_errors and not color2_errors:
            msg = "Must configure flux errors for at least color1 or color2."
            raise FieldValidationError(self.__class__.color1_flux1_err, self, msg)

    @property
    def columns(self):
        columns = (self.color1_flux1,
                   self.color1_flux2,
                   self.color2_flux1,
                   self.color2_flux2)

        if self.color1_flux1_err:
            # Config validation ensures if one is set, both are set.
            columns = columns + (self.color1_flux1_err,
                                 self.color1_flux2_err)

        if self.color2_flux1_err:
            # Config validation ensures if one is set, both are set.
            columns = columns + (self.color2_flux1_err,
                                 self.color2_flux2_err)

        return columns

    def __call__(self, df):
        k = 2.5/np.log(10.)

        color1_flux1 = df[self.color1_flux1].values*u.Unit(self.color1_flux1_units)
        color1_mag1 = color1_flux1.to(u.ABmag).value
        if self.color1_flux1_err:
            color1_mag1_err = k*df[self.color1_flux1_err].values/df[self.color1_flux1].values
        else:
            color1_mag1_err = 0.0

        color1_flux2 = df[self.color1_flux2].values*u.Unit(self.color1_flux2_units)
        color1_mag2 = color1_flux2.to(u.ABmag).value
        if self.color1_flux2_err:
            color1_mag2_err = k*df[self.color1_flux2_err].values/df[self.color1_flux2].values
        else:
            color1_mag2_err = 0.0

        color2_flux1 = df[self.color2_flux1].values*u.Unit(self.color2_flux1_units)
        color2_mag1 = color2_flux1.to(u.ABmag).value
        if self.color2_flux1_err:
            color2_mag1_err = k*df[self.color2_flux1_err].values/df[self.color2_flux1].values
        else:
            color2_mag1_err = 0.0

        color2_flux2 = df[self.color2_flux2].values*u.Unit(self.color2_flux2_units)
        color2_mag2 = color2_flux2.to(u.ABmag).value
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


class AstromDiff(MultiColumnAction):
    """Calculate the difference between two columns, assuming their units
    are degrees, and convert the difference to arcseconds.

    Parameters
    ----------
    df : `pandas.core.frame.DataFrame`
        The catalog to calculate the position difference from.

    Returns
    -------
    The difference.

    Notes
    -----
    The columns need to be in units (specifiable in
    the radecUnits1 and 2 config options) that can be converted
    to arcseconds. This action doesn't have any calibration
    information and assumes that the positions are already
    calibrated.
    """

    col1 = Field(doc="Column to subtract from", dtype=str)
    radecUnits1 = Field(doc="Units for col1", dtype=str, default="degree")
    col2 = Field(doc="Column to subtract", dtype=str)
    radecUnits2 = Field(doc="Units for col2", dtype=str, default="degree")
    returnMilliArcsecs = Field(doc="Use marcseconds or not?", dtype=bool, default=True)

    @property
    def columns(self):
        return (self.col1, self.col2)

    def __call__(self, df):
        angle1 = df[self.col1].values * u.Unit(self.radecUnits1)

        angle2 = df[self.col2].values * u.Unit(self.radecUnits2)

        angleDiff = angle1 - angle2

        if self.returnMilliArcsecs:
            angleDiffValue = angleDiff.to(u.arcsec) * 1000
        else:
            angleDiffValue = angleDiff.value
        return angleDiffValue
