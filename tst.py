import numpy as np

# EZ-Readerrun
from Scripts.helper_functions import load_parameters_dict as load_parameters
from EZReader.Corpus import corpus
from EZReader.Model import run_EZReader

# load corpus for EZ-Reader
corpus = corpus("Data/SRC98Corpus.txt")

# load parameters
parameters = load_parameters("Data/parameters.txt")

# start model
fixations = run_EZReader(parameters, corpus, NRuns=1, sentences=[5])

print(fixations)
