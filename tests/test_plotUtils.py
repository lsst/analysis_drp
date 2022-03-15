import unittest
import numpy as np

from lsst.analysis.drp.plotUtils import stellarLocusFit, perpDistance
import lsst.utils.tests


class FitTest(lsst.utils.tests.TestCase):
    """Test to see if the fitting and distance calculations are working"""

    def testFitLine(self):
        """Make a known array of points for x and y and then test that
        the derived fit parameters are as expected."""

        xs = np.arange(1, 10)
        ys = np.arange(1, 10)

        # Define an initial fit box that encompasses the points
        testParams = {"xMin": 0, "xMax": 11, "yMin": 0, "yMax": 11, "mHW": 1, "bHW": 0}
        paramsOut = stellarLocusFit(xs, ys, testParams)

        # stellarLocusFit performs two iterations of fitting and also
        # calculates the perpendicular gradient to the fit line and
        # the points of intersection between the box and the fit
        # line. Test that these are returning what is expected.
        self.assertFloatsAlmostEqual(paramsOut["mODR"], 1.0)
        self.assertFloatsAlmostEqual(paramsOut["mODR2"], 1.0)
        self.assertFloatsAlmostEqual(paramsOut["bODR"], 0.0)
        self.assertFloatsAlmostEqual(paramsOut["bODR2"], 0.0)
        self.assertFloatsAlmostEqual(paramsOut["bPerpMin"], 0.0)
        self.assertFloatsAlmostEqual(paramsOut["bPerpMax"], 22.0)

    def testPerpDistance(self):
        """Test the calculation of the perpendicular distance"""

        p1 = np.array([1, 1])
        p2 = np.array([2, 2])
        testPoints = np.array([[1, 2], [1.5, 1.5], [2, 1]])
        # perpDistance uses two points, p1 and p2, to define a line
        # then calculates the perpendicular distance of the testPoints
        # to this line
        dists = perpDistance(p1, p2, testPoints)
        self.assertFloatsAlmostEqual(np.array([1.0/np.sqrt(2), 0.0, -1.0/np.sqrt(2)]), np.array(dists))


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
