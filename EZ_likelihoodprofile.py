# modules
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import sys

# EZ-Reader
from EZReader.Corpus import corpus
from EZReader.Model import run_EZReader
from Scripts.helper_functions import load_parameters_dict as load_parameters
from Scripts.helper_functions import mkdir_p
from Scripts.Annotate import initiate_corpus as read_corpus
from Scripts.Annotate import annotate_reading_data
from Scripts import Metrics
from Likelihood_synth_Wood.synthetic_Likelihood import calc_likelihood


def EZ_loglik(parval):
    # make deep copies because of parallelism vs. pass-by-reference
    local_parameters = deepcopy(parameters)
    local_text_array = deepcopy(text_array)
    local_corpus = deepcopy(corpus)
    local_parameters[parameter_name] = parval
    #fixations = run_EZReader(local_parameters, local_corpus, local_text_array, NRuns=2000, timeout=300)
    fixations = run_EZReader(local_parameters, local_corpus, NRuns=20, timeout=300)
    loglik = calc_loglik(fixations)
    return loglik


def calc_loglik(fixations):
    # assume that s is called stat_s and is global
    # assume also that totalNumberOfWords is global
    # annotate y
    y_star = annotate_reading_data(fixations)
    groups = y_star.groupby("runN")
    s_star = groups.apply(this_metric, ancorp).to_frame(
        "stat").reset_index()
    return calc_likelihood(s_star[["stat"]], stat_s)


# number of evaluations per parameter
N_positions = int(sys.argv[3])

# define all parameter ranges
par_ranges = {
    # controls the overall rate of lexical processing
    'Alpha1': np.linspace(start=50, stop=250, num=N_positions),
    # factor for saccade length
    'Eta2': np.linspace(start=0, stop=0.5, num=N_positions),
    # determines the absolute amount by which eccentricity modulates the slowing effect of limited visual acuity
    'Epsilon': np.linspace(start=1, stop=1.5, num=N_positions),
    # modulates the probability of initiating a corrective saccade
    'Lambda': np.linspace(start=0, stop=1.5, num=N_positions),
    # word frequency effect
    'Alpha2': np.linspace(start=0.0, stop=10.0, num=N_positions),
    # cloze predictability effect
    'Alpha3': np.linspace(start=0.0, stop=50.0, num=N_positions),
    # lexical access (familiarity check * Delta = lexical access)
    # fairly enthusiastic range
    'Delta': np.linspace(start=0.1, stop=1.5, num=N_positions),
    # post-lexical integration dur
    # Reichle investigated this Parameter in the 2009 article in the range of 0-200
    'I': np.linspace(start=10.0, stop=100.0, num=N_positions),
    # attention shift dur
    'A': np.linspace(start=10.0, stop=100.0, num=N_positions),
    # labile saccade dur
    'M1': np.linspace(start=75.0, stop=200.0, num=N_positions),
    # non-labile saccade dur
    # given that Reichle always says E-Z Reader simulates skilled readers and that M2=25 is quite high in the scale, 10-100 is a reasonable range
    'M2': np.linspace(start=10.0, stop=100.0, num=N_positions),
    # saccade execution dur
    'S': np.linspace(start=10.0, stop=100.0, num=N_positions),
    # factor for engage time of current M1 to adjust new M1 duration accordingly
    'Xi': np.linspace(start=0.0, stop=1.0, num=N_positions),
    # random saccadic error
    # when =1 huge random errors accure (-6.28 in a saccade with intended length of 10 characters), maybe restrict even more
    'Eta1': np.linspace(start=0.1, stop=1.0, num=N_positions),
    # systematic saccadic error
    # =10 gives no systematic error for saccades with intended length of 10 characters, maybe include up to 10 then?
    'Psi': np.linspace(start=1.0, stop=9.0, num=N_positions),
    # systematic saccadic error again
    # when holding the other parameters constant this range is equal to a systematic error of roughly +2.3 - -2.7 characters in saccades with intended lentgh of 10 characters
    'Omega1': np.linspace(start=3.0, stop=8.0, num=N_positions),
    # systematic saccadic error once again / factor for launch site fixation duration
    'Omega2': np.linspace(start=1.5, stop=10.0, num=N_positions),
    # pre-attentive visual processing time
    # given that Reichle simulated skilled readers with V=50
    'V': np.linspace(start=10.0, stop=150.0, num=N_positions),
    # sigma whenever something is taken from a gamma distribution
    'sigmaGamma': np.linspace(start=3.0, stop=200.0, num=N_positions),
    # probability of integration failure
    'pF': np.linspace(start=0.0, stop=0.2, num=N_positions)  # =0.2 is huge
}

# choose parameter
parameter_name = sys.argv[1]
# parameter_name = "Alpha1"
LL_table = pd.DataFrame({'parameters': par_ranges.get(parameter_name)})

# define all metrics
metrics = {
    1: Metrics.calc_FMFD,
    2: Metrics.calc_FMFD_sd,
    3: Metrics.calc_gazeDur,
    4: Metrics.calc_gazeDur_sd,
    5: Metrics.calc_meanFixPerSentence,
    6: Metrics.calc_meanFixPerSentence_sd,
    7: Metrics.calc_refixProb,
    8: Metrics.calc_SFD,
    9: Metrics.calc_SFD_sd,
    10: Metrics.calc_SFP,
    11: Metrics.calc_SMFD,
    12: Metrics.calc_SMFD_sd,
    13: Metrics.calc_TVT,
    14: Metrics.calc_TVT_sd
    # 13: Metrics.calc_SP,
    # : Metrics.calc_refixProb_sd,
    # : Metrics.calc_SFP_sd,
    # : Metrics.calc_SP_sd,
}


# load corpus for analysis
ancorp = read_corpus("Data/SRC98Corpus.txt")

# load dataset y
y = pd.read_csv("Data/singlerun.txt", sep="\t")
y = annotate_reading_data(y)  # calculate statistics of base data

# choose metric
this_metric = metrics.get(int(sys.argv[2]))
# this_metric = metrics.get(1)
metric_name = this_metric.__name__
metric_name = metric_name.rsplit("calc_")[1]
stat_s = this_metric(y, ancorp)

# load corpus for EZ-Reader
corpus = corpus("Data/SRC98Corpus.txt")

# load parameters
parameters = load_parameters("Data/parameters.txt")

outfile_metricsfirst = metric_name + "_" + parameter_name
outfile_parsfirst = parameter_name + "_" + metric_name

LL_table['sLL'] = LL_table.applymap(lambda x: EZ_loglik(x))

# save data
LL_table.to_csv("Data/LLresults_" + outfile_metricsfirst + ".txt", sep="\t")


# create plot numero uno
plt.figure(figsize=(8, 5))

plt.title("Log(synthetic likelihood) of " +
          metric_name, fontdict={"fontweight": "bold"})

# also automate the labeling of the x axis through the files?
plt.xlabel(parameter_name)
plt.ylabel("Log Synthetic Likelihood")

# plt.ylim([-50.5, 5.5])  # same ranges for y-axis

# plot a black vertical line where the "true" value of the Parameter is used for generating y -> s
plt.plot(LL_table.parameters, LL_table.sLL, "ro", marker=".", c="r", alpha=0.7)
plt.axvline(x=parameters.get(parameter_name),
            ymin=0,
            ymax=1,
            c="black")


plotpath1 = "Plots_parsfirst"
plotpath2 = "Plots_metricsfirst"
mkdir_p(plotpath1)
mkdir_p(plotpath2)
plt.savefig(plotpath1 + "/LLprofil_" + outfile_parsfirst + ".png", dpi=300)
plt.savefig(plotpath2 + "/LLprofil_" + outfile_metricsfirst + ".png", dpi=300)


# create plot numero dos
plt.figure(figsize=(8, 5))

plt.title("synthetic likelihood of " +
          metric_name, fontdict={"fontweight": "bold"})

# also automate the labeling of the x axis through the files?
plt.xlabel(parameter_name)
plt.ylabel("Synthetic Likelihood")

# plt.ylim([-50.5, 5.5])  # same ranges for y-axis

# plot a black vertical line where the "true" value of the Parameter is used for generating y -> s
plt.plot(LL_table.parameters, np.exp(LL_table.sLL),
         "ro", marker=".", c="r", alpha=0.7)
plt.axvline(x=parameters.get(parameter_name),
            ymin=0,
            ymax=1,
            c="black")

plt.savefig("Plots_parsfirst/Lprofil_" + outfile_parsfirst + ".png", dpi=300)
plt.savefig("Plots_metricsfirst/Lprofil_" +
            outfile_metricsfirst + ".png", dpi=300)
