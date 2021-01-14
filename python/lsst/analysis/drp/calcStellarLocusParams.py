import numpy as np
import scipy.odr as scipyODR

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig


class CalcStellarLocusParamsTaskConnections(pipeBase.PipelineTaskConnections,
                                            dimensions=("tract", "skymap"),
                                            defaultTemplates={"inputCoaddName": "deep", "fitType": "wFit"}):

    catPlot = pipeBase.connectionTypes.Input(doc="The tract wide catalog to make plots from.",
                                             storageClass="DataFrame",
                                             name="qaTable_tract",
                                             dimensions=("tract", "skymap"))

    fitParams = pipeBase.connectionTypes.Output(doc="The parameters from the fit to the stellar locus.",
                                                storageClass="StructuredDataDict",
                                                name="stellarLocusParams_{fitType}",
                                                dimensions=("tract", "skymap"))

    skymap = pipeBase.connectionTypes.Input(doc="The skymap for the tract",
                                            storageClass="SkyMap",
                                            name="{inputCoaddName}Coadd_skyMap",
                                            dimensions=("skymap",))


class CalcStellarLocusParamsTaskConfig(pipeBase.PipelineTaskConfig,
                                       pipelineConnections=CalcStellarLocusParamsTaskConnections):

    xColName = pexConfig.Field(
        doc="The column name for the values to be plotted on the x axis.",
        dtype=str,
        default="coord_ra",
    )

    yColName = pexConfig.Field(
        doc="The column name for the values to be plotted on the y axis.",
        dtype=str,
        default="coord_dec",
    )

    sourceTypeColName = pexConfig.Field(
        doc="The column to use for star - galaxy separation.",
        dtype=str,
        default="iExtendedness",
    )

    name = pexConfig.Field(
        doc="The name of the subject of the plot.",
        dtype=str,
        default="wFit",
    )


class CalcStellarLocusParamsTask(pipeBase.PipelineTask):

    ConfigClass = CalcStellarLocusParamsTaskConfig
    _DefaultName = "CalcStellarLocusParamsTask"

    def run(self, catPlot, skymap):

        fitParams = self.calcStellarLocusParams(catPlot)

        return pipeBase.Struct(fitParams=fitParams)

    def hardwiredFits(self, name):
        """Hardwired fits taken from pipe_analysis"""

        # Straight line fits for the perpendicular ranges
        # The following fits were derived in the process of calibrating
        # the above coeffs (all three RC2 tracts gave ~ the same fits).
        # May remove later if deemed no longer useful.

        if name == "wFit":
            return 0.52, -0.08
        elif name == "xFit":
            return 13.35, -15.54
        elif name == "yFit":
            return 0.40, 0.03

    def calcStellarLocusParams(self, catPlot):
        """Fit a line to the stellar locus

        Parameters
        ----------
        catPlot : `pandas.core.frame.DataFrame`

        Returns
        -------
        paramDict : `dict`
            The parameters of the fit to the stellar locus.
                ``"bHW"``
                    The hardwired intercept to fall back on.
                ``"b_odr"``
                    The intercept calculated by the orthogonal distance
                    regression fitting.
                ``"mHW"``
                    The hardwired gradient to fall back on.
                ``"m_odr"``
                    The gradient calculated by the orthogonal distance
                    regression fitting.
                ``"magLim"``
                    The magnitude limit used in the fitting.
                ``"x1`"``
                    The x minimum of the box used in the fit.
                ``"x2"``
                    The x maximum of the box used in the fit.
                ``"y1"``
                    The y minimum of the box used in the fit.
                ``"y2"``
                    The y maximum of the box used in the fit.

        Notes
        -----
        Uses scipy's orthogonal distance regression to fit a linear function
        to the given area of the stellar locus. The box used to fit in is
        determined by the config option, `name`. The sources are then cut down
        using the useForQAFlag added by the addQAColumns task and then by
        magnitude (< 22.0). Stars are separated from galaxies using the
        `self.config.sourceTypeColName` column.
        """

        # Get the limits of the box used for the statistics
        # Putting random values in here for now
        self.log.info(("Fitting: the values of {} against {}".format(
                       self.config.xColName, self.config.yColName)))

        if self.config.name == "yFit":
            xBoxLims = (0.82, 2.01)
            yBoxLims = (0.37, 0.9)
        elif self.config.name == "wFit":
            xBoxLims = (0.1, 1.0)
            yBoxLims = (0.02, 0.48)
        elif self.config.name == "xFit":
            xBoxLims = (1.05, 1.55)
            yBoxLims = (0.78, 1.62)

        mHW, bHW = self.hardwiredFits(self.config.name)

        # Cut the catalogue down to only valid sources
        catPlot = catPlot[catPlot["useForQAFlag"].values]

        # Only use sources brighter than something
        # TODO: what is that something
        catPlot = catPlot[(catPlot["iCModelMag"] < 22.0)]

        # Need to separate stars and galaxies
        stars = (catPlot[self.config.sourceTypeColName] == 0.0)

        # For stars
        xsStars = catPlot[self.config.xColName].values[stars]
        ysStars = catPlot[self.config.yColName].values[stars]

        # Points to use for the fit
        fitPoints = np.where((xsStars > xBoxLims[0]) & (xsStars < xBoxLims[1])
                             & (ysStars > yBoxLims[0]) & (ysStars < yBoxLims[1]))[0]

        def linearFit(B, x):
            return B[0]*x + B[1]

        linear = scipyODR.Model(linearFit)
        linear = scipyODR.polynomial(1)

        data = scipyODR.Data(xsStars[fitPoints], ysStars[fitPoints])
        odr = scipyODR.ODR(data, linear, beta0=[bHW, mHW])
        params = odr.run()
        params.pprint()

        paramDict = {"mHW": float(mHW), "bHW": float(bHW), "m_odr": float(params.beta[1]),
                     "b_odr": float(params.beta[0]), "x1": xBoxLims[0], "x2": xBoxLims[1],
                     "y1": yBoxLims[0], "y2": yBoxLims[1], "magLim": 22.0}

        return paramDict
