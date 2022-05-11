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

import unittest

import galsim
import numpy as np

import lsst.utils.tests
from lsst.afw.geom import Quadrupole
from lsst.analysis.drp.calcFunctors import CalcE, CalcE1, CalcE2, CalcShapeSize


class ShapeSizeTestCase(lsst.utils.tests.TestCase):
    """Test ellipiticity and size calculations."""

    @classmethod
    def setUpClass(cls):
        cls.data = np.array(
            [
                (1.3, 1.3, 0.0),  # e1 = e2 = 0
                (2.4, 1.2, 0.6),  # e1 = e2 != 0
                (1.0, 2.0, 0.0),  # e1 < 0; e2 = 0
                (3.5, 3.5, 0.5),  # e1 = 0; e2 > 0
                (3.0, 1.5, -1.2),  # e1 > 0; e2 < 0
            ],
            dtype=[("ixx", "<f8"), ("iyy", "<f8"), ("ixy", "<f8")],
        )

    def test_size(self):
        """Test CalcShapeSize functor"""
        traceSize = CalcShapeSize(sizeType="trace")(self.data)
        determinantSize = CalcShapeSize(sizeType="determinant")(self.data)

        for idx, row in enumerate(self.data):
            shape = Quadrupole(ixx=row["ixx"], iyy=row["iyy"], ixy=row["ixy"])
            self.assertFloatsAlmostEqual(traceSize[idx], shape.getTraceRadius(), rtol=1e-8)
            self.assertFloatsAlmostEqual(determinantSize[idx], shape.getDeterminantRadius(), rtol=1e-8)
            # Arithmetic mean >= Geometric mean implies that
            # trace radius is never smaller than determinant radius.
            self.assertGreaterEqual(traceSize[idx], determinantSize[idx])

    def test_complex_shear(self):
        """Test CalcE functor

        Test that our ellipticity calculation under the two conventions are
        accurate by comparing with GalSim routines.
        """
        shear = CalcE(ellipticityType="epsilon")(self.data)
        distortion = CalcE(ellipticityType="chi")(self.data)
        size = CalcShapeSize(sizeType="determinant")(self.data)
        for idx, row in enumerate(self.data):
            galsim_shear = galsim.Shear(shear[idx])
            self.assertFloatsAlmostEqual(distortion[idx].real, galsim_shear.e1)
            self.assertFloatsAlmostEqual(distortion[idx].imag, galsim_shear.e2)
            # Check that the ellipiticity values correspond to the moments.
            A = galsim_shear.getMatrix() * size[idx]
            M = np.dot(A.transpose(), A)
            self.assertFloatsAlmostEqual(M[0, 0], row["ixx"], rtol=1e-8)
            self.assertFloatsAlmostEqual(M[1, 1], row["iyy"], rtol=1e-8)
            self.assertFloatsAlmostEqual(M[0, 1], row["ixy"], rtol=1e-8)

    def test_halve_angle(self):
        """Test ``halvePhaseAngle`` parameter in CalcE

        Test that setting ``halvePhaseAngle`` to True halves the phase angle
        while keeping the magnitude the same.
        """
        ellip = CalcE(ellipticityType="epsilon")(self.data)
        ellip_half = CalcE(ellipticityType="epsilon", halvePhaseAngle=True)(self.data)
        self.assertFloatsAlmostEqual(np.abs(ellip), np.abs(ellip_half))

        for idx, row in enumerate(self.data):
            galsim_shear = galsim.Shear(ellip[idx])
            galsim_shear_half = galsim.Shear(ellip_half[idx])
            self.assertFloatsAlmostEqual(np.abs(ellip_half[idx]), galsim_shear.g)
            self.assertFloatsAlmostEqual(
                galsim_shear.beta / galsim.radians, 2 * galsim_shear_half.beta / galsim.radians
            )

    @lsst.utils.tests.methodParameters(ellipticityType=("chi", "epsilon"))
    def test_shear_components(self, ellipticityType):
        """Test CalcE1 and CalcE2 functors

        This test checks if CalcE1 and CalcE2 correspond to the real and
        imaginary components of CalcE.
        """
        ellip = CalcE(ellipticityType=ellipticityType)(self.data)
        e1 = CalcE1(ellipticityType=ellipticityType)(self.data)
        e2 = CalcE2(ellipticityType=ellipticityType)(self.data)

        self.assertFloatsAlmostEqual(np.real(ellip), e1)
        self.assertFloatsAlmostEqual(np.imag(ellip), e2)

    def test_e1_validation(self):
        """Test that CalcE1 throws an exception when misconfigured."""
        CalcE1(ellipticityType="chi", colXy=None).validate()
        with self.assertRaises(ValueError):
            CalcE1(ellipticityType="epsilon", colXy=None).validate()

    def test_size_validation(self):
        """Test that CalcShapeSize throws an exception when misconfigured."""
        CalcShapeSize(sizeType="trace", colXy=None).validate()
        with self.assertRaises(ValueError):
            CalcShapeSize(sizeType="determinant", colXy=None).validate()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
