# This class contains various methods for displaying (in the output file) various corpus and eye-movement data.
import numpy as np

# **************************************************************************

# display trace of fixations (detailed information):

# instead of a file, create a dictionary with the relevant metrics of the fixation trace:


def traceList(fixationList, sentence, S, trace_array, subject):
    for i in range(len(trace_array)):
        if trace_array[i].word is None:
            continue
        fixation = [trace_array[i].number, trace_array[i].dur, trace_array[i].pos, trace_array[i].word, sentence.word(trace_array[i].word).frequency, int(np.around(sentence.word(trace_array[i].word).frequency_class)), S, subject]
        fixationList.append(fixation)  # update the dictionary with fixation-data with index-variable as key
