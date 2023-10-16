# EZ-Reader
from Scripts.helper_functions import load_parameters_dict as load_parameters

from EZReader.Corpus import corpus

from EZReader.Model import run_EZReader

# load corpus for EZ-Reader
corpus = corpus("Data/SRC98Corpus.txt")

# load parameters
parameters = load_parameters("Data/parameters.txt")

fixations = run_EZReader(parameters, corpus, NRuns=500, timeout=300)

#fixations.to_csv("s_500.txt", sep="\t")
