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
import lsst.utils.tests

from lsst.analysis.drp.calcFunctors import MagDiff
from lsst.analysis.drp.dataSelectors import GalaxyIdentifier
from lsst.analysis.drp.plotUtils import get_and_remove_figure_text
from lsst.analysis.drp.scatterPlot import ScatterPlotWithTwoHistsTask, ScatterPlotWithTwoHistsTaskConfig

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import os
import pandas as pd
import shutil
import tempfile

matplotlib.use("Agg")

ROOT = os.path.abspath(os.path.dirname(__file__))
filename_texts_ref = os.path.join(ROOT, "data", "test_scatterPlot_texts.txt")
path_lines_ref = os.path.join(ROOT, "data", "test_scatterPlot_lines")


class ScatterPlotWithTwoHistsTaskTestCase(lsst.utils.tests.TestCase):
    """ScatterPlotWithTwoHistsTask test case."""
    def setUp(self):
        self.testDir = tempfile.mkdtemp(dir=ROOT, prefix="test_output")

        # Set up a quasi-plausible measurement catalog
        mag = 12.5 + 2.5*np.log10(np.arange(10, 100000))
        flux = 10**(-0.4*(mag - (mag[-1] + 1)))
        rng = default_rng(0)
        extendedness = 0. + (rng.uniform(size=len(mag)) < 0.99*(mag - mag[0])/(mag[-1] - mag[0]))
        flux_meas = flux + rng.normal(scale=np.sqrt(flux*(1 + extendedness)))
        flux_err = np.sqrt(flux_meas * (1 + extendedness))
        good = (flux_meas/np.sqrt(flux * (1 + extendedness))) > 3
        extendedness = extendedness[good]
        flux = flux[good]
        flux_meas = flux_meas[good]
        flux_err = flux_err[good]

        # Configure the plot to show observed vs true mags
        config = ScatterPlotWithTwoHistsTaskConfig(
            axisLabels={"x": "mag", "y": "mag meas - ref", "mag": "mag"},
        )
        config.selectorActions.flagSelector.bands = ["i"]
        config.axisActions.yAction = MagDiff(col1="refcat_flux", col2="refcat_flux")
        config.nonBandColumnPrefixes.append("refcat")
        config.sourceSelectorActions.galaxySelector = GalaxyIdentifier
        config.highSnStatisticSelectorActions.statSelector.threshold = 50
        config.lowSnStatisticSelectorActions.statSelector.threshold = 20
        self.task = ScatterPlotWithTwoHistsTask(config=config)

        n = len(flux)
        self.bands, columns = config.get_requirements()
        data = {
            "refcat_flux": flux,
            "patch": np.zeros(n, dtype=int),
        }

        # Assign values to columns based on their unchanged default names
        for column in columns:
            if column not in data:
                if column.startswith("detect"):
                    data[column] = np.ones(n, dtype=bool)
                elif column.endswith("_flag") or "Flag" in column:
                    data[column] = np.zeros(n, dtype=bool)
                elif column.endswith("Flux"):
                    config.axisActions.yAction.col1 = column
                    data[column] = flux_meas
                elif column.endswith("FluxErr"):
                    data[column] = flux_err
                elif column.endswith("_extendedness"):
                    data[column] = extendedness
                else:
                    raise RuntimeError(f"Unexpected column {column} in ScatterPlotWithTwoHistsTaskConfig")

        self.data = pd.DataFrame(data)

    def tearDown(self):
        if os.path.exists(self.testDir):
            shutil.rmtree(self.testDir, True)
        del self.bands
        del self.data
        del self.task

    def test_ScatterPlotWithTwoHistsTask(self):
        plt.rcParams.update(plt.rcParamsDefault)
        result = self.task.run(self.data,
                               dataId={},
                               runName="test",
                               skymap=None,
                               tableName="test",
                               bands=self.bands,
                               plotName="test")

        self.assertTrue(isinstance(result.scatterPlot, plt.Figure))

        # Set to true to save plots as PNGs
        # Use matplotlib.testing.compare.compare_images if needed
        save_images = False
        if save_images:
            result.scatterPlot.savefig(os.path.join(ROOT, "data", "test_scatterPlot.png"))

        texts, lines = get_and_remove_figure_text(result.scatterPlot)
        if save_images:
            result.scatterPlot.savefig(os.path.join(ROOT, "data", "test_scatterPlot_unlabeled.png"))

        # Set to true to re-generate reference data
        resave = False

        # Compare line values
        for idx, line in enumerate(lines):
            filename = os.path.join(path_lines_ref, f"line_{idx}.txt")
            if resave:
                np.savetxt(filename, line)
            arr = np.loadtxt(filename)
            # Differences of order 1e-12 possible between MacOS and Linux
            # Plots are generally not expected to be that precise
            # Differences to 1e-3 should not be visible with this test data
            self.assertFloatsAlmostEqual(arr, line, atol=1e-3, rtol=1e-4)

        # Ensure that newlines within labels are replaced by a sentinel
        newline = '\n'
        newline_replace = "[newline]"
        # Compare text labels
        if resave:
            with open(filename_texts_ref, 'w') as f:
                f.writelines(f'{text.strip().replace(newline, newline_replace)}\n' for text in texts)

        with open(filename_texts_ref, 'r') as f:
            texts_ref = set(x.strip() for x in f.readlines())
        texts_set = set(x.strip().replace(newline, newline_replace) for x in texts)

        self.assertGreaterEqual(texts_set, texts_ref)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
