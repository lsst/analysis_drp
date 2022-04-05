# -*- python -*-
from lsst.sconsUtils import scripts

# Python-only package
scripts.BasicSConstruct(
    "analysis_drp",
    disableCc=True,
    noCfgFile=True,
    defaultTargets=scripts.DEFAULT_TARGETS + ("pipelines",),
)
