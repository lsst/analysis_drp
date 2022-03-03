__all__ = ["maskPixelsPercentCalc"]

import numpy as np


def maskPixelsPercentCalc(image, maskName):

    dim = image.image.getDimensions()
    ntot = dim.x * dim.y

    maskBit = image.mask.getMaskPlaneDict()[maskName]

    # How many pixels are masked by this particular mask bit?
    nmask = np.count_nonzero(image.mask.array.flatten() & 2**maskBit)

    return 100.*nmask/ntot
