# This class instantiates the random- and systematic-error components of saccades.

import random as r
import numpy as np


class Saccade:

    ################################
    def __init__(self, IntendedSaccadeLength, launchSiteFixDur):
        self.IntendedSaccadeLength = IntendedSaccadeLength
        self.launchSiteFixDur = launchSiteFixDur

    def randomError(self, PAR):
        result = r.gauss(0.0, 1.0) * (PAR["Eta1"] + (np.abs(self.IntendedSaccadeLength) * PAR["Eta2"]))
        return result

    ################################
    def systematicError(self, PAR):
        result = (PAR["Psi"] - np.abs(self.IntendedSaccadeLength)) * ((PAR["Omega1"] - np.log(self.launchSiteFixDur)) / PAR["Omega2"])
        if self.IntendedSaccadeLength < 0:
            result *= -1
        return result
