# This file is part of analysis_drp.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import annotations

import argparse
import re
import sys
from typing import Iterable, Mapping, Optional, Sequence

from lsst.resources import ResourcePath
from lsst.pipe.base.pipelineIR import PipelineIR, LabeledSubset


class PlotPipelineBuilder:
    """Class to support building a complete analysis drp pipeline from an input
    file.
    """

    def __init__(self):
        # Setup the command line argument parser that will be used to build
        # the pipeline
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("pipeline", help="input pipeline to process")
        self.parser.add_argument(
            "subsetName",
            nargs="?",
            help="name of the subset which will contain all the tasks defined in the input file, leave "
            "blank if no subset is to be created",
        )

    def getSubsets(self) -> Mapping[str, tuple[Optional[str], Iterable[str]]]:
        """Allows defining subsets based on pattern matching labels.

        In order to define a subset, it should be added to the mapping returned
        by this method. The key will be the label used for the subset, and the
        value is a list of regular expressions that will be used to match the
        labels in the input pipeline.

        For instance returning:
        >>> {'axisAxis': ["ScatterPlot_.*", ".*HistPlot.*", "SpecificPlot"]}

        Will create a subset named axisAxis, and will contain any label with
        "ScatterPlot_" at the beginning of its label, "HistPlot" anywhere in
        the label, and exactly the label "SpecificPlot".
        """
        return {}

    def run(self, args: Sequence[str]) -> None:
        """Main entry point for actually building a pipeline. This is
        responsible for parsing the supplied args (likely coming from the
        command line), and then calling the build method. The command expects
        that args contain the script name in the 0th index.
        """
        parsedArgs = self.parser.parse_args(args[1:])
        resource = ResourcePath(parsedArgs.pipeline)

        # validate that the supplied files are considered inputs and not
        # finalized pipelines
        if resource.getExtension() != ".in":
            raise ValueError("This builder can only process pipeline files that end in .in")
        self.build(resource, parsedArgs.subsetName)

    @property
    def excludeSet(self) -> set[str]:
        """Returns a list of labels which should be ignored for some reason"""
        return set()

    def build(self, pipeline: ResourcePath, subsetName: Optional[str] = None) -> None:
        """Builds a pipeline out of the input proto-pipeline supplied to run.

        Subsets are added to the output pipeline that contain all the plots
        (except plots contained in the exclude set), and any other subsets
        defined within the builder's ``getSubsets`` method.

        Parameters
        ----------
        pipeline : `ResourcePath`
            The `ResourcePath` that points to an input proto-pipeline that is
            to be transformed into a final pipeline.

        subsetName : `str`, optional
            If supplied, a subset will be created containing all the tasks in
            the input proto-pipeline file with the supplied name. Defaults to
            `None`.

        Raises
        ------
        ValueError
            Raised if a subset is already declared with a name returned from
            the ``getGroups`` method.
            Raised if a subset is already declared with a name that conflicts
            with the subset name for all analysis plots.
        """
        inputPipeline = PipelineIR.from_uri(pipeline)
        excludes = self.excludeSet
        for name, (description, subset_exps) in self.getSubsets():
            tmpTaskSet = set()
            for exp in subset_exps:
                pattern = re.compile(exp)
                for task in inputPipeline.tasks:
                    if task in excludes:
                        continue
                    if pattern.match(task):
                        tmpTaskSet.add(task)
            if tmpTaskSet:
                if name in inputPipeline.labeled_subsets:
                    raise ValueError(
                        f"Conflicting subset name: {name} is already " "declared as a labeled subset"
                    )
                inputPipeline.labeled_subsets[name] = LabeledSubset(
                    label=name, subset=tmpTaskSet, description=description
                )
        # need to exclude any labels that should not be in the pipeline as
        # defined in the excluded labels set
        for label in self.excludeSet:
            inputPipeline.tasks.pop(label)

        # add a generic labeled subset with all tasks in it
        if subsetName:
            if subsetName in inputPipeline.labeled_subsets:
                raise ValueError(
                    f"Labeled Subset {subsetName} was already declared in input "
                    "pipeline, please ensure this label is not defined in any "
                    "input pipelines"
                )
            inputPipeline.labeled_subsets[subsetName] = LabeledSubset(
                label=subsetName,
                subset=set(inputPipeline.tasks) - self.excludeSet,
                description=None,
            )

        inputPipeline.write_to_uri(pipeline.updatedExtension("yaml"))


if __name__ == "__main__":
    PlotPipelineBuilder().run(sys.argv)
