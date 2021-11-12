# modules
import numpy as np
import pandas as pd

# EZ-Reader
from Scripts.helper_functions import load_parameters_dict as load_parameters
from EZReader.Corpus import corpus as Corpus
from EZReader.Model import run_EZReader

# Metrics & synthetic Likelihood
from Scripts import Metrics
from Scripts.Annotate import annotate_reading_data, initiate_corpus as read_corpus
from Likelihood_synth_Wood.synthetic_Likelihood import calc_likelihood

# pyDREAM
from DREAM_mod.PyDREAM_master.pydream.core import run_dream
from DREAM_mod.PyDREAM_master.pydream.parameters import SampledParam
from scipy.stats import norm as normal, truncnorm as truncated_normal
from copy import deepcopy


# short function for a truncated normal prior distribution bounded between lb and ub with a sd of (ub-lb)/2 and a mean at the centre of the range


def trunc_gauss(lower, upper):
    mean = lower + (upper - lower) / 2.
    sd = (upper - lower) / 2.
    a, b = (lower - mean) / sd, (upper - mean) / sd
    return SampledParam(truncated_normal, a=a, b=b, scale=sd, loc=mean)


def EZ_loglik(parvals):
    # make deep copies because of parallelism vs. pass-by-reference
    local_parameters = deepcopy(parameters)
    local_corpus = deepcopy(corpus)
    for i in range(len(parvals)):
        pname = list(par_ranges.keys())[i]
        if np.isinf(par_ranges[pname].prior(parvals[i])):
            print("Trying to set %s to %lf which is out of bounds!" % (pname, parvals[i]))
            return -np.Inf
        # if pname == "Eta2":
        #     local_parameters["Eta2"] = np.exp(parvals[i])
        local_parameters[pname] = parvals[i]
    fixations = run_EZReader(local_parameters, local_corpus, NRuns=EZ_runs)
    loglik = calc_loglik(fixations)
    return loglik


def calc_loglik(fixations):
    # assume that s is called stat_s and is global
    # assume also that totalNumberOfWords is global
    # annotate y
    y_star = annotate_reading_data(fixations)
    groups = y_star.groupby("runN")

    s_GD = groups.apply(Metrics.calc_gazeDur, ancorp).to_frame("GD")
    s_MFPS = groups.apply(Metrics.calc_meanFixPerSentence, ancorp).to_frame("MFPS")

    s_star = s_GD.join(s_MFPS)
    return calc_likelihood(s_star, stat_s)


# end of definitions


# load corpus for analysis
ancorp = read_corpus("Data/SRC98Corpus.txt")

# load dataset y
y = pd.read_csv("Data/singlerun.txt", sep="\t")  # load table with entries separated by a space (" ")
y = annotate_reading_data(y)  # calculate statistics of base data

# calculate data stats
stat_s = [Metrics.calc_gazeDur(y, ancorp), Metrics.calc_meanFixPerSentence(y, ancorp)]
print(stat_s)

# load corpus
corpus = Corpus("Data/SRC98Corpus.txt")

# load parameters
parameters = load_parameters("Data/parameters.txt")

par_ranges = {
    # 'Alpha1': trunc_gauss(50, 250),
    'Delta': trunc_gauss(0.1, 1.5),
    'Epsilon': trunc_gauss(1, 1.5),
    'Lambda': trunc_gauss(0, 1.5),
    'M1': trunc_gauss(75, 200),
    'Omega2': trunc_gauss(1, 10),
    'V': trunc_gauss(10, 150),
    'Xi': trunc_gauss(0, 1)
}


rand = np.random
niter = 10000
nchains = 30
EZ_runs = 10
outfile = "SynthL_8pars_long"


# start DREAM
sampled_params, log_ps = run_dream(list(par_ranges.values()), EZ_loglik, nchains=nchains, nseedchains=70, niterations=niter, restart=False, verbose=True, model_name=outfile, stochastic_loglike=True, multiprocessing=True)

# save output
for i in range(nchains):
    with open("%s_%.2d.dat" % (outfile, i + 1), "w") as f:
        f.write("\t".join(list(par_ranges.keys()) + ["loglik"]) + "\n")
        for j in range(niter):
            f.write("\t".join(str(x) for x in sampled_params[i][j, :]))
            f.write("\t" + str(log_ps[i][j, 0]) + "\n")
        f.flush()
        f.close()
