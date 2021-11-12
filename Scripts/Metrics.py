# Contains all functions used to deduce statistics s from data y.

import numpy as np

# +++++++++++++++++++++++++++++++++++++
# Durations
# +++++++++++++++++++++++++++++++++++++


def calc_SFD(x, corpus):
    # calculates word-based mean of single fixation durations
    # requires annotated fixation sequences (ONE run (.groupby(["runN"]))) AND given corpus

    # identify single fixations:
    cond = (x["fixationN"] != 0) & (x["cnfix1"] == 1) & (x["xnfix1"] == 1)

    # create a new column with SFDuration if condition applies or NaN where not
    x["SFD"] = np.where(cond, x["duration"], float("NaN"))

    # merge corpus and x to get df corpus with additional column "SFD" which contains NaN, or actual SFD
    joined = corpus.merge(x, on=["sentenceN", "wordN"], how='left')

    # calculate mean of SMFs:
    out = joined["SFD"].mean()

    # return value and df
    return out


def calc_SFD_sd(x, corpus):
    # calculates standard deviation of single fixation durations
    # requires annotated fixation sequences (ONE run (.groupby(["runN"]))) AND given corpus

    # identify single fixations:
    cond = (x["fixationN"] != 0) & (x["cnfix1"] == 1) & (x["xnfix1"] == 1)

    # create a new column with SFDuration if condition applies or NaN where not
    x["SFD"] = np.where(cond, x["duration"], float("NaN"))

    # merge corpus and x to get df corpus with additional column "SFD" which contains NaN, or actual SFD
    joined = corpus.merge(x, on=["sentenceN", "wordN"], how='left')

    # calculate sd of SFDs:
    out = joined["SFD"].std()

    # return value and df
    return out


def calc_FMFD(x, corpus):
    # calculates word-based mean of first of multiple fixation durations
    # requires annotated fixation sequences (ONE run (.groupby(["runN"]))) AND given corpus

    # identify first fixation when there are multiple:
    cond = (x["fixationN"] != 0) & (x["cnfix1"] >= 2) & (x["xnfix1"] == 1)

    # create a new column with FMFDuration if condition applies or NaN where not
    x["FMFD"] = np.where(cond, x["duration"], float("NaN"))

    # merge corpus and x to get df corpus with additional column "FMFD" which contains NaN, or actual FMFD
    joined = corpus.merge(x, on=["sentenceN", "wordN"], how='left')

    # calculate mean of SMFs:
    out = joined["FMFD"].mean()

    # return value and df
    return out


def calc_FMFD_sd(x, corpus):
    # calculates standard deviation of first of multiple fixation durations
    # requires annotated fixation sequences (ONE run (.groupby(["runN"]))) AND given corpus

    # identify first fixation when there are multiple:
    cond = (x["fixationN"] != 0) & (x["cnfix1"] >= 2) & (x["xnfix1"] == 1)

    # create a new column with FMFDuration if condition applies or NaN where not
    x["FMFD"] = np.where(cond, x["duration"], float("NaN"))

    # merge corpus and x to get df corpus with additional column "FMFD" which contains NaN, or actual FMFD
    joined = corpus.merge(x, on=["sentenceN", "wordN"], how='left')

    # calculate sd of FMFDs:
    out = joined["FMFD"].std()

    # return value and df
    return out


def calc_SMFD(x, corpus):
    # calculates word-based mean of second of multiple fixation durations
    # requires annotated fixation sequences (ONE run (.groupby(["runN"]))) AND given corpus

    # identify second fixation when there are multiple:
    cond = (x["fixationN"] != 0) & (x["cnfix1"] >= 2) & (x["xnfix1"] == 2)

    # create a new column with SMFDuration if condition applies or NaN where not
    x["SMFD"] = np.where(cond, x["duration"], float("NaN"))

    # merge corpus and x to get df corpus with additional column "SMFD" which contains NaN, or actual SMFD
    joined = corpus.merge(x, on=["sentenceN", "wordN"], how='left')

    # calculate mean of SMFDs:
    out = joined["SMFD"].mean()

    # return value and df
    return out


def calc_SMFD_sd(x, corpus):
    # calculates standard deviation of second of multiple fixation durations
    # requires annotated fixation sequences (ONE run (.groupby(["runN"]))) AND given corpus

    # identify second fixation when there are multiple:
    cond = (x["fixationN"] != 0) & (x["cnfix1"] >= 2) & (x["xnfix1"] == 2)

    # create a new column with SMFDuration if condition applies or NaN where not
    x["SMFD"] = np.where(cond, x["duration"], float("NaN"))

    # merge corpus and x to get df corpus with additional column "SMFD" which contains NaN, or actual SMFD
    joined = corpus.merge(x, on=["sentenceN", "wordN"], how='left')

    # calculate sd of SMFDs:
    out = joined["SMFD"].std()

    # return value and df
    return out


def calc_TVT(x, corpus):
    # calculates word-based total viewing time
    # requires annotated fixation sequences (ONE run (.groupby(["runN"]))) AND given corpus

    # create a groupby-object of the fixations grouped by sentenceN and by wordN
    # sum the individual fixation durations
    TVT = x.groupby(["sentenceN", "wordN"])["duration"].sum()

    # join corpus and TVT, so that for every word in every sentence there is the TVT in column "duration"
    joined = corpus.merge(TVT, on=["sentenceN", "wordN"], how='left')

    # calculate mean of TVT
    out = joined["duration"].mean()

    return out


def calc_TVT_sd(x, corpus):
    # calculates sd of word-based total viewing time
    # requires annotated fixation sequences (ONE run (.groupby(["runN"]))) AND given corpus

    # create a groupby-object of the fixations grouped by sentenceN and by wordN
    # sum the individual fixation durations
    TVT = x.groupby(["sentenceN", "wordN"])["duration"].sum()

    # join corpus and TVT, so that for every word in every sentence there is the TVT in column "duration"
    joined = corpus.merge(TVT, on=["sentenceN", "wordN"], how='left')

    # calculate sd of TVT
    out = joined["duration"].std()

    return out


def calc_gazeDur(x, corpus):
    # calculates word-based gaze duration
    # requires annotated fixation sequences (ONE run (.groupby(["runN"]))) AND given corpus

    # have only firstpass fixations in df "firstPass"
    firstPass = x.loc[(x["firstpass"] == True)]

    # create a groupby-object of the fixations grouped by sentenceN and by wordN
    # sum the individual fixation durations
    gazeDur = firstPass.groupby(["sentenceN", "wordN"])["duration"].sum()

    # join corpus and gazeDur, so that for every word in every sentence there is the gaze duration in column "duration"
    joined = corpus.merge(gazeDur, on=["sentenceN", "wordN"], how='left')

    # calculate mean of gaze durations
    out = joined["duration"].mean()

    return out


def calc_gazeDur_sd(x, corpus):
    # calculates standard deviation of word-based gaze duration
    # requires annotated fixation sequences (ONE run (.groupby(["runN"]))) AND given corpus

    # have only firstpass fixations in df "firstPass"
    firstPass = x.loc[(x["firstpass"] == True)]

    # create a groupby-object of the fixations grouped by sentenceN and by wordN
    # sum the individual fixation durations
    gazeDur = firstPass.groupby(["sentenceN", "wordN"])["duration"].sum()

    # join corpus and gazeDur, so that for every word in every sentence there is the gaze duration in column "duration"
    joined = corpus.merge(gazeDur, on=["sentenceN", "wordN"], how='left')

    # calculate sd of gaze durations
    out = joined["duration"].std()

    return out


# +++++++++++++++++++++++++++++++++++++
# Probabilities
# +++++++++++++++++++++++++++++++++++++

def calc_SFP(x, corpus):
    # calculates word-based probability of single fixation
    # requires annotated fixation sequences (ONE run (.groupby(["runN"]))) AND given corpus

    # identify single fixations / condition:
    cond = (x["fixationN"] != 0) & (x["cnfix1"] == 1) & (x["xnfix1"] == 1)

    # have =1 everywhere condition applies and =0 where not
    x["SFP"] = np.where(cond, 1, 0)

    # merge corpus and x to get df corpus with additional column "SFP" which contains NaN, 0, or actual skip per word
    # joined = corpus.set_index(["sentenceN", "wordN"]).join(x.set_index(["sentenceN", "wordN"]))
    joined = corpus.merge(x, on=["sentenceN", "wordN"], how='left')

    # calculate overall probability for single fixations:
    out = joined["SFP"].mean()

    # return value
    return out


def calc_SFP_sd(x, corpus):
    # calculates standard deviation of single fixation probability
    # requires annotated fixation sequences (ONE run (.groupby(["runN"]))) AND given corpus

    # identify single fixations / condition:
    cond = (x["fixationN"] != 0) & (x["cnfix1"] == 1) & (x["xnfix1"] == 1)

    # have =1 everywhere condition applies and =0 where not
    x["SFP"] = np.where(cond, 1, 0)

    # merge corpus and x to get df corpus with additional column "SFP" which contains NaN, 0, or actual skip per word
    # joined = corpus.set_index(["sentenceN", "wordN"]).join(x.set_index(["sentenceN", "wordN"]))
    joined = corpus.merge(x, on=["sentenceN", "wordN"], how='left')

    # calculate sd of single fixations probability:
    out = joined["SFP"].std()

    # return value
    return out


def calc_SP(x, corpus):
    # calculates word-based skipping probability
    # requires annotated fixation sequences (ONE run (.groupby(["runN"]))) AND given corpus

    # have only firstpass fixations in df "firstPass"
    firstPass = x.loc[(x["firstpass"] == True)]

    # create a copy of column "wordN" with name "skip"
    firstPass["skip"] = firstPass["wordN"]

    # condition for skipping words
    cond = firstPass.skip.shift(-1) - firstPass.skip >= 2

    # have =1 everywhere condition applies and =0 where not
    firstPass["skip"] = np.where(cond, 1, 0)

    # count amount of skips on each word
    skips = firstPass.groupby(["sentenceN", "wordN"])["skip"].sum()
    skips = skips.reset_index()  # have the columns "sentenceN", "wordN", and "skip" in df skips

    # convert values above 1.0 to 1.0:
    skips["skip"].values[skips["skip"].values > 1] = 1

    # join corpus and skips to get df corpus with additional column "skip" which contains NaN, 0, or actual skip per
    # word
    joined = corpus.set_index(["sentenceN", "wordN"]).join(skips.set_index(["sentenceN", "wordN"]))

    # calculate overall skipping probability:
    out = joined["skip"].mean()

    # return value
    return out


def calc_SP_sd(x, corpus):
    # calculates sd of skipping probability
    # requires annotated fixation sequences (ONE run (.groupby(["runN"]))) AND given corpus

    # have only firstpass fixations in df "firstPass"
    firstPass = x.loc[(x["firstpass"] == True)]

    # condition for skipping words
    cond = firstPass.wordN.shift(-1) - firstPass.wordN >= 2

    # have =1 everywhere condition applies and =0 where not
    firstPass["skip"] = np.where(cond, 1, 0)

    # count amount of skips on each word
    skips = firstPass.groupby(["sentenceN", "wordN"])["skip"].sum()
    skips = skips.reset_index()  # have the columns "sentenceN", "wordN", and "skip" in df skips

    # convert values above 1.0 to 1.0:
    skips["skip"].values[skips["skip"].values > 1] = 1

    # join corpus and skips to get df corpus with additional column "skip" which contains NaN, 0, or actual skip per
    # word
    joined = corpus.set_index(["sentenceN", "wordN"]).join(skips.set_index(["sentenceN", "wordN"]))

    # calculate sd of skipping probability:
    out = joined["skip"].std()

    # return value
    return out


# gives corpus with appropiate refixProb for each individual word and the overall refixProb (mean)
def calc_refixProb(x, corpus):
    # calculates word-based probability of refixation
    # requires annotated fixation sequences (ONE run (.groupby(["runN"]))) AND given corpus

    # have only firstpass fixations in df "firstPass"
    firstPass = x.loc[(x["firstpass"] == True)]

    # when in two rows (fixations) we get the same wordN, we have a refixation:
    firstPass["refixation"] = firstPass.wordN.eq(firstPass.wordN.shift(-1))
    # firstPass["refixation"] = np.greater_equal(firstPass.cnfix1, 2)  # if cnfix1 >= 2: refixation = True
    # Kann es sein, dass in der Methode mit cnfix1 mehr True rauskommen? (erste Fixation auf Wort ebenfalls als True)

    # convert booleans for refixation to integers:
    firstPass["refixProb"] = firstPass["refixation"].astype(int)

    # count amount of refixations on each word
    refixNum = firstPass.groupby(["sentenceN", "wordN"])["refixProb"].sum()
    refixNum = refixNum.reset_index()  # have the columns "sentenceN", "wordN", and "refixProb" in df refixNum

    # convert values above 1.0 to 1.0:
    refixNum["refixProb"].values[refixNum["refixProb"].values > 1] = 1

    # join corpus and refixNum to get df corpus with additioanl column "refixProb" which contains NaN, 0,
    # or actual refixProb per word
    joined = corpus.set_index(["sentenceN", "wordN"]).join(refixNum.set_index(["sentenceN", "wordN"]))

    # calculate overall refixation probability:
    out = joined["refixProb"].mean()

    # return value
    return out


def calc_refixProb_sd(x, corpus):
    # calculates sd of refixation probability
    # requires annotated fixation sequences (ONE run (.groupby(["runN"]))) AND given corpus

    # have only firstpass fixations in df "firstPass"
    firstPass = x.loc[(x["firstpass"] == True)]

    # when in two rows (fixations) we get the same wordN, we have a refixation:
    firstPass["refixation"] = firstPass.wordN.eq(firstPass.wordN.shift(-1))
    # firstPass["refixation"] = np.greater_equal(firstPass.cnfix1, 2)  # if cnfix1 >= 2: refixation = True
    # Kann es sein, dass in der Methode mit cnfix1 mehr True rauskommen? (erste Fixation auf Wort ebenfalls als True)

    # convert booleans for refixation to integers:
    firstPass["refixProb"] = firstPass["refixation"].astype(int)

    # count amount of refixations on each word
    refixNum = firstPass.groupby(["sentenceN", "wordN"])["refixProb"].sum()
    refixNum = refixNum.reset_index()  # have the columns "sentenceN", "wordN", and "refixProb" in df refixNum

    # convert values above 1.0 to 1.0:
    refixNum["refixProb"].values[refixNum["refixProb"].values > 1] = 1

    # join corpus and refixNum to get df corpus with additioanl column "refixProb" which contains NaN, 0,
    # or actual refixProb per word
    joined = corpus.set_index(["sentenceN", "wordN"]).join(refixNum.set_index(["sentenceN", "wordN"]))

    # calculate sd of refixation probability:
    out = joined["refixProb"].std()

    # return value
    return out


# +++++++++++++++++++++++++++++++++++++
# Other
# +++++++++++++++++++++++++++++++++++++

def calc_meanFixPerSentence(x, dummy):
    # calculates the mean number of fixations per sentence

    # create a groupby-object of the fixations grouped by sentenceN
    sentence = x.groupby(["sentenceN"])

    # count the number of fixations per groupby-object
    nrOfFix = sentence["fixationN"].count()

    # calculate mean of the number of fixations for every sentence
    out = nrOfFix.mean()

    return out


def calc_meanFixPerSentence_sd(x, dummy):
    # calculates the standard deviation of number of fixations per sentence

    # create a groupby-object of the fixations grouped by sentenceN
    sentence = x.groupby(["sentenceN"])

    # count the number of fixations per groupby-object
    nrOfFix = sentence["fixationN"].count()

    # calculate sd of the number of fixations for every sentence
    out = nrOfFix.std()

    return out
