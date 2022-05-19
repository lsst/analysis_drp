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


__all__ = ("BinnedCorr2Config",)

from lsst.pex.config import ChoiceField, Config, Field, FieldValidationError


class BinnedCorr2Config(Config):
    """A Config class that holds the various parameters supported by treecorr.

    The fields in this class correspond to the parameters that can be passed to
    BinnedCorr2 in `treecorr`, which is the base class for all two-point
    correlation function calculations. The default values set for the fields
    are identical to the default values set in v4.2 of `treecorr`. The
    parameters that are excluded in this class are
    'verbose', 'log_file', 'output_dots', 'rng' and 'pairwise' (deprecated).
    For details about these options, see the documentation for `treecorr`:
    https://rmjarvis.github.io/TreeCorr/_build/html/correlation2.html

    A separate config class is used instead
    of constructing a `~lsst.pex.config.DictField` so that mixed types can be
    supported and the config can be validated at the beginning of the
    execution. The ``toDict`` provides a minimal dictionary that override only
    the default values and excludes the key-values pairs when the item is None.

    Notes
    -----
    This is intended to be used in CalcRhoStatistics class.
    """

    nbins = Field(
        doc=(
            "How many bins to use. "
            "(Exactly three of nbins, bin_size, min_sep, max_sep "
            "are required. If nbins is not given, it will be "
            "calculated from the values of the other three, "
            "rounding up to the next highest integer. "
            "In this case, bin_size will be readjusted to account "
            "for this rounding up."
        ),
        dtype=int,
        optional=True,
        check=lambda x: x > 0,
    )

    bin_size = Field(
        doc=(
            "The width of the bins in log(separation). "
            "Exactly three of nbins, bin_size, min_sep, max_sep are required. "
            "If bin_size is not given, it will be calculated from the values "
            "of the other three."
        ),
        dtype=float,
        optional=True,
    )

    min_sep = Field(
        doc=(
            "The minimum separation in units of sep_units, if relevant. "
            "Exactly three of nbins, bin_size, min_sep, max_sep are required. "
            "If min_sep is not given, it will be calculated from the values "
            "of the other three."
        ),
        dtype=float,
        optional=True,
    )

    max_sep = Field(
        doc=(
            "The maximum separation in units of sep_units, if relevant. "
            "Exactly three of nbins, bin_size, min_sep, max_sep are required. "
            "If max_sep is not given, it will be calculated from the values "
            "of the other three."
        ),
        dtype=float,
        optional=True,
    )

    sep_units = ChoiceField(
        doc=(
            "The units to use for the separation values, given as a string. "
            "This includes both min_sep and max_sep above, as well as the "
            "units of the output distance values."
        ),
        default="radian",
        dtype=str,
        optional=True,
        allowed={
            units: units
            for units in ["arcsec", "arcmin", "degree", "hour", "radian"]
        },
    )

    bin_slop = Field(
        doc=(
            "How much slop to allow in the placement of pairs in the bins. "
            "If bin_slop = 1, then the bin into which a particular pair is "
            "placed may be incorrect by at most 1.0 bin widths. "
            r"If None, use a bin_slop that gives a maximum error of 10% on "
            "any bin, which has been found to yield good results for most "
            "applications."
        ),
        default=None,
        dtype=float,
        optional=True,
    )

    brute = Field(
        doc=(
            "Whether to use the 'brute force' algorithm? "
            "Unlike treecorr, setting this to 1 or 2 is not supported."
        ),
        default=False,
        dtype=bool,
        optional=True,
    )

    split_method = ChoiceField(
        doc=("How to split the cells in the tree when building the tree " "structure."),
        default="mean",
        dtype=str,
        optional=True,
        allowed={
            "mean": "Use the arithmetic mean of the coordinate being split.",
            "median": "Use the median of the coordinate being split.",
            "middle": (
                "Use the middle of the range; i.e. the average of the "
                "minimum and maximum value."
            ),
            "random": (
                "Use a random point somewhere in the middle two "
                "quartiles of the range."
            ),
        },
    )

    min_top = Field(
        doc=(
            "The minimum number of top layers to use when setting up the "
            "field.  The top-level cells are where each calculation job "
            "starts."
        ),
        default=3,
        dtype=int,
        optional=True,
    )

    max_top = Field(
        doc=(
            "The maximum number of top layers to use when setting up the "
            "field. The top-level cells are where each calculation job "
            "starts. There will typically be of order 2**max_top cells."
        ),
        default=10,
        dtype=int,
        optional=True,
    )

    precision = Field(
        doc=(
            "The precision to use for the output values. This specifies how "
            "many digits to write."
        ),
        default=4,
        dtype=int,
        optional=True,
        check=lambda x: x > 0,
    )

    m2_uform = ChoiceField(
        doc="The default functional form to use for aperture mass calculations.",
        default="Crittenden",
        dtype=str,
        optional=True,
        allowed={
            "Crittenden": "Crittenden et al. (2002); ApJ, 568, 20",
            "Schneider": "Schneider, et al (2002); A&A, 389, 729",
        },
    )

    metric = ChoiceField(
        doc=(
            "Which metric to use for distance measurements. For details, see "
            "https://rmjarvis.github.io/TreeCorr/_build/html/metric.html"
        ),
        default="Euclidean",
        dtype=str,
        optional=True,
        allowed={
            "Euclidean": "straight-line Euclidean distance between two points",
            "FisherRperp": (
                "the perpendicular component of the distance, "
                "following the definitions in "
                "Fisher et al, 1994 (MNRAS, 267, 927)"
            ),
            "OldRperp": (
                "the perpendicular component of the distance using the "
                "definition of Rperp from TreeCorr v3.x."
            ),
            "Rlens": (
                "Distance from the first object (taken to be a lens) to "
                "the line connecting Earth and the second object "
                "(taken to be a lensed source)."
            ),
            "Arc": "the true great circle distance for spherical coordinates.",
            "Periodic": "Like ``Euclidean``, but with periodic boundaries.",
        },
    )

    bin_type = ChoiceField(
        doc="What type of binning should be used?",
        default="Log",
        dtype=str,
        optional=True,
        allowed={
            "Log": (
                "Logarithmic binning in the distance. The bin steps will "
                "be uniform in log(r) from log(min_sep) .. log(max_sep)."
            ),
            "Linear": (
                "Linear binning in the distance. The bin steps will be "
                "uniform in r from min_sep .. max_sep."
            ),
            "TwoD": (
                "2-dimensional binning from x = (-max_sep .. max_sep) "
                "and y = (-max_sep .. max_sep). The bin steps will be "
                "uniform in both x and y. (i.e. linear in x,y)"
            ),
        },
    )

    min_rpar = Field(
        doc=(
            "The minimum difference in Rparallel to allow for pairs being "
            "included in the correlation function. "
        ),
        default=None,
        dtype=float,
        optional=True,
    )

    max_rpar = Field(
        doc=(
            "The maximum difference in Rparallel to allow for pairs being "
            "included in the correlation function. "
        ),
        default=None,
        dtype=float,
        optional=True,
    )

    period = Field(
        doc="For the 'Periodic' metric, the period to use in all directions.",
        default=None,
        dtype=float,
        optional=True,
    )

    xperiod = Field(
        doc="For the 'Periodic' metric, the period to use in the x direction.",
        default=None,
        dtype=float,
        optional=True,
    )

    yperiod = Field(
        doc="For the 'Periodic' metric, the period to use in the y direction.",
        default=None,
        dtype=float,
        optional=True,
    )

    zperiod = Field(
        doc="For the 'Periodic' metric, the period to use in the z direction.",
        default=None,
        dtype=float,
        optional=True,
    )

    var_method = ChoiceField(
        doc="Which method to use for estimating the variance",
        default="shot",
        dtype=str,
        optional=True,
        allowed={
            method: method
            for method in [
                "shot",
                "jackknife",
                "sample",
                "bootstrap",
                "marked_bootstrap",
            ]
        },
    )

    num_bootstrap = Field(
        doc=(
            "How many bootstrap samples to use for the 'bootstrap' and "
            "'marked_bootstrap' var methods."
        ),
        default=500,
        dtype=int,
        optional=True,
    )

    num_threads = Field(
        doc=(
            "How many OpenMP threads to use during the calculation. "
            "If set to None, use all the available CPU cores."
        ),
        default=None,
        dtype=int,
        optional=True,
    )

    def validate(self):
        # Docs inherited from base class
        super().validate()
        req_params = (self.nbins, self.bin_size, self.min_sep, self.max_sep)
        num_req_params = sum(param is not None for param in req_params)
        if num_req_params != 3:
            msg = (
                "You must specify exactly three of ``nbins``, ``bin_size``, ``min_sep`` and ``max_sep``"
                f" in treecorr_config. {num_req_params} parameters were set instead."
            )
            raise FieldValidationError(self.__class__.bin_size, self, msg)

        if self.min_sep is not None and self.max_sep is not None:
            if self.min_sep > self.max_sep:
                raise FieldValidationError(
                    self.__class__.min_sep, self, "min_sep must be <= max_sep"
                )

        if self.min_rpar is not None and self.max_rpar is not None:
            if self.min_rpar > self.max_rpar:
                raise FieldValidationError(
                    self.__class__.min_rpar, self, "min_rpar must be <= max_rpar"
                )

    def toDict(self):
        # Docs inherited from base class
        # We are excluding items that are None due to treecorr limitations.
        # TODO: DM-38480. This override can be removed after treecorr v4.3 is
        # released and makes its way onto rubin-env.
        return {k: v for k, v in super().toDict().items() if v is not None}
