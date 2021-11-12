# Contains all functions which are needed to annotate data and corpus (like Reichle uses in E-Z Reader 10).

import pandas as pd
import numpy as np


def annotate_reading_data(x):
    # adds columns for first pass, xnfix1, xnfix2, cnfix1, cnfix2 to fixation sequences
    # expects a pandas dataframe
    # requires column "subjectN" (to be replaced with runN)
    # requires column "wordN"
    # requires column "sentenceN"

    # add first pass marker
    x["CM"] = x.groupby(["runN", "sentenceN"]).wordN.apply(lambda a: a.cummax())
    x["D"] = x["wordN"] - x["CM"]
    x["firstpass"] = x.groupby(["runN", "sentenceN", "CM"]).D.apply(lambda a: a.cummin() == 0)
    x = x.drop(columns=["CM", "D"])

    # count first pass fixations
    x["cnfix1"] = x.groupby(["runN", "sentenceN", "wordN"]).firstpass.transform('sum')

    # count second pass fixations
    x["cnfix2"] = x.groupby(["runN", "sentenceN", "wordN"]).firstpass.transform('count')
    x["cnfix2"] = x["cnfix2"] - x["cnfix1"]

    # enumerate first pass fixations
    x["xnfix1"] = 0
    fp_ix = x["firstpass"] == True
    x.loc[fp_ix, "xnfix1"] = x[fp_ix].groupby(["runN", "sentenceN", "wordN"]).cumcount() + 1

    # enumerate second + more pass fixations
    x["xnfix2"] = 0
    fp_ix = x["firstpass"] == False
    x.loc[fp_ix, "xnfix2"] = x[fp_ix].groupby(["runN", "sentenceN", "wordN"]).cumcount() + 1

    return x


# needed in initiate_corpus function
def add_wn(x):
    # requires groupby-object of corpus (groupby(sentenceN)) with column "wordN"
    # give every word(line) the appropiate "sn"- and "wn-values

    x["wordN"] = np.arange(x.shape[0]) + 1
    # !!!# "+1" wird später benötigt, wenn Modell Output angepasst
    # Output ist angepasst...
    return x


def initiate_corpus(x):
    # requires path to corpus file (x)
    # corpus file has to have layout used by Reichle E-Z Reader Java application
    # gives out adjusted pandas DataFrame (ready to be used by functions which give probability metrics)

    # read corpus file
    corpus = pd.read_csv(x, header=None, delim_whitespace=True)

    # give columns meaningful names
    corpus = corpus.rename(columns={0: "freq", 1: "wlen", 2: "cloze", 3: "word"})
    # "freq": frequency
    # "wlen": word length
    # "cloze": cloze predictability
    # "word": actual spelled out word

    # use the "@" to count and enumerate sentences
    corpus["sentenceN"] = corpus.word.str.contains("@").shift(periods=1, fill_value=0).cumsum() + 1
    # !!!# "+1" wird später noch gebraucht, wenn der Modell Output angepasst wurde (beginnt mit 1 zu zählen anstatt mit 0...)
    # +1 wird jetzt gebraucht, sonst gibt es Konflikte mit den Metriken wo der corpus als Variable mitgegeben wird

    # remove the "@", because it's annoying
    corpus["wordN"] = corpus.word.str.replace("@", "")

    # give every word(line) the appropriate "sn"- and "wn-values
    out = corpus.groupby("sentenceN").apply(add_wn)

    return out
