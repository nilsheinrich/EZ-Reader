# This class instantiates all of EZ Reader's component processes (e.g., L1).

import random as r
import numpy as np
from EZReader.Gamma import nextDouble as gdist


class processing:
    def __init__(self):
        self.dur: float = 0.0  # duration in ms
        self.durCopy: float = 0.0  # Copy of duration is retained (i.e., not decremented)
        self.name: str = "None"  # process Label (e.g., "L1", "M2", etc.)
        self.wn: int = 0  # word associated with process (e.g., L1 for word nr. 5)

    # **************************************************************************
    # ADJUST LEXICAL-PROCESSING RATE
    # **************************************************************************

    def calcRate(self, PAR, word, currentPos):
        # Calculate mean absolute deviation between fixation position & letters of attended word:
        meanAbsDev = 0
        for i in range(word.pos1, word.posN + 1):
            meanAbsDev += np.abs(i - currentPos)
        meanAbsDev /= word.length

        # Return updated lexical-processing rate:
        return np.power(PAR["Epsilon"], meanAbsDev)

    # **************************************************************************
    # PRE-ATTENTIVE VISUAL PROCESSING
    # **************************************************************************


class initializeV(processing):
    def __init__(self, PAR, N):
        self.name = "V"
        self.wn = N

        # calculate duration (ms):
        self.dur = PAR["V"]

    # **************************************************************************
    # FAMILIARITY CHECK
    # **************************************************************************


class initializeL1(processing):
    def __init__(self, PAR, word, ongoingI):
        self.name = "L1"
        self.wn = word.wn

        # determine if word can be predicted (i.e., has the previous word been integrated?):
        clozeValue: float
        if ongoingI:
            clozeValue = 0
        else:
            clozeValue = 1.0

        # calculate duration (ms):
        PrGuess: float = r.random()
        if PrGuess < word.cloze:
            self.dur = 0
        else:
            mu = PAR["Alpha1"] - (PAR["Alpha2"] * word.nlog_frequency) - (PAR["Alpha3"] * (word.cloze * clozeValue))
            self.dur = gdist(mu, PAR["SigmaGamma"])

    # **************************************************************************
    # LEXICAL ACCESS
    # **************************************************************************


class initializeL2(processing):
    def __init__(self, PAR, word, ongoingI):
        self.name = "L2"
        self.wn = word.wn

        # Determine if word can be predicted (i.e., has the previous word been integrated?):
        clozeValue = 0
        if ongoingI == False:
            clozeValue = 1

        # Calculate duration (ms):
        mu = PAR["Alpha1"] - (PAR["Alpha2"] * word.nlog_frequency) - (PAR["Alpha3"] * (word.cloze * clozeValue))
        mu *= PAR["Delta"]
        self.dur = gdist(mu, PAR["SigmaGamma"])

    # **************************************************************************
    # POST-LEXICAL INTEGRATION
    # **************************************************************************


class initializeI(processing):
    def __init__(self, PAR, N):
        self.name = "I"
        self.wn = N

        # Calculate duration of I (ms):
        self.dur = gdist(PAR["I"], PAR["SigmaGamma"])

        # In this version this functions does not differentiate between target words and non-target words
        # (which Reichle does in the original)

    # **************************************************************************
    # ATTENTION SHIFT
    # **************************************************************************


class initializeA(processing):
    def __init__(self, PAR, N):
        self.name = "A"
        self.wn = N

        # Calculate duration (ms):
        self.dur = gdist(PAR["A"], PAR["SigmaGamma"])

    # **************************************************************************
    # LABILE SACCADIC PROGRAMMING
    # **************************************************************************


class initializeM1(processing):
    def __init__(self, PAR, activeProcesses, currentPos, targetPos, N):
        self.name = "M1"
        self.wn = N

        # Calculate intended saccade length:
        self.IntendedSaccadeLength = targetPos - currentPos

        # Calculate duration (ms):
        self.dur = gdist(PAR["M1"], PAR["SigmaGamma"])
        self.durCopy = self.dur

        # Cancel any pending M1 & adjust new M1 duration accordingly:
        for i in reversed(range(len(activeProcesses))):
            if activeProcesses[i].name == "M1":
                programmingTimeCompleted = activeProcesses[i].durCopy - activeProcesses[i].dur
                engageTime = self.dur * PAR["Xi"]
                if programmingTimeCompleted > engageTime:
                    self.dur -= engageTime
                else:
                    self.dur -= programmingTimeCompleted
                del activeProcesses[i]

    # **************************************************************************
    # NON-LABILE SACCADIC PROGRAMMING
    # **************************************************************************


class initializeM2(processing):
    def __init__(self, PAR, N, saccade):
        self.name = "M2"
        self.wn = N

        # Calculate saccadic error (character spaces):
        randomError = saccade.randomError(PAR)
        systematicError = saccade.systematicError(PAR)

        # Calculate actual saccade length (character spaces):
        self.ActualSaccadeLength = saccade.IntendedSaccadeLength + randomError + systematicError

        # Calculate duration (ms):
        self.dur = gdist(PAR["M2"], PAR["SigmaGamma"])

    # **************************************************************************
    # SACCADE EXECUTION
    # **************************************************************************


class initializeS(processing):
    def __init__(self, PAR, ActualSaccadeLength, N):
        self.name = "S"
        self.wn = N

        # Calculate length (character spaces):
        self.ActualSaccadeLength = ActualSaccadeLength

        # Calculate duration (ms):
        self.dur = PAR["S"]
