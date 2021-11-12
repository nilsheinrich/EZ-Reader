import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import scipy.stats as st
# from scipy.stats.stats import pearsonr  # for correlation coefficient
import pymc3  # for heatmaps -> causes Warning
import itertools

from Scripts.helper_functions import load_parameters_dict as load_parameters

# load parameters
parameters = load_parameters("Data/parameters.txt")

# get files
mypath = ""
files = [str(f) for f in Path(mypath).iterdir() if f.match("SynthL_8pars_testing_*.dat")]

# load all the data into a single dataframe:
df = pd.DataFrame()
for file in files:
    chain = float(file[21:-4])
    data = pd.read_csv(file, sep="\t")  # reads the df and separating the columns by space
    data["chain"] = chain
    df = pd.concat([df, data])  # puts both df together to have the M1 value now in every appropriate column

# get dataFrames:
pars = df.columns.to_list()  # get column labels in list
pars = pars[0: -2]  # exclude loglik and chain labels

par_combinations = list(itertools.combinations(pars, 2))  # get every combination of two elements from list "pars"

##################
# plotting
# KDE's:
for parameter in pars:
    # parameter:
    data = df[f"{parameter}"]
    data = np.asarray(data)

    # compute hpdi:
    hpdi_bounds = pymc3.stats.hpd(data, 0.25)  # I went for the smallest interval which contains 25% of the mass

    # plot boundaries:
    lbound = min(data)
    ubound = max(data)

    # instantiate KDE
    data_kde_Epsilon = np.linspace(lbound, ubound, 100)
    kde_Epsilon = st.gaussian_kde(data)

    # Grid
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"KDE {parameter}", fontdict={"fontweight": "bold"})

    # axis labels
    ax.set_xlabel("Parameter Values")
    ax.set_ylabel("Density")

    plt.axvline(parameters[f"{parameter}"], 0, 100, c="black")
    # plot a black vertical line where the "true" value of the Parameter is used for generating y -> s

    ax.set_xlim([lbound, ubound])

    xaxis = np.linspace(lbound, ubound, 10)
    ax.set_xticks(xaxis)

    # Plotting
    ax.plot(data_kde_Epsilon, kde_Epsilon(data_kde_Epsilon), color="r")

    # HPDI:
    ax.axvspan(hpdi_bounds[0], hpdi_bounds[1], alpha=0.3, color="b")
    point_estimate = (hpdi_bounds[0] + hpdi_bounds[1]) / 2
    plt.axvline(point_estimate, 0, 100, c="b")

    plt.savefig(f"Density {parameter}", dpi=300)
    plt.close()

# Heatmaps
for combination in par_combinations:
    parameter_x = combination[0]
    parameter_y = combination[1]

    N_bins = 100

    # Grid
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Correlation heatmap", fontdict={"fontweight": "bold"})

    # Construct 2D histogram from data using the 'plasma' colormap
    ax.hist2d(df[parameter_x], df[parameter_y], bins=N_bins, density=False, cmap='plasma')

    # Plot a colorbar with label.
    # cb = plt.colorbar()
    # cb.set_label('Density')

    # Add title and labels to plot.
    # plt.title("Correlation heatmap")
    ax.set_xlabel(parameter_x)
    ax.set_ylabel(parameter_y)

    plt.savefig(f"Correlation between {parameter_x} and {parameter_y} - Heatmap", dpi=300)
    plt.close()

# Caterpillars
for parameter in pars:
    # data = df[f"{parameter}"]
    data = df

    # initiating Plot:
    plt.figure(figsize=(8, 5))

    plt.title(f"Caterpillar {parameter}", fontdict={"fontweight": "bold"})

    plt.xlabel("Dream iteration")
    plt.ylabel("Parameter Value")

    Values = np.arange(0, 10001, 1000)  # parameterValues labeled on x axis
    plt.xticks(Values)  # which values should appear in x axis

    # chain01:
    chain_data = data.loc[data["chain"] == 1]
    plt.plot(chain_data.index, chain_data[f"{parameter}"], linestyle="solid", c="red", alpha=0.7, linewidth=0.4)
    # chain02:
    chain_data = data.loc[data["chain"] == 2]
    plt.plot(chain_data.index, chain_data[f"{parameter}"], linestyle="solid", c="blue", alpha=0.7, linewidth=0.4)
    # chain03:
    chain_data = data.loc[data["chain"] == 3]
    plt.plot(chain_data.index, chain_data[f"{parameter}"], linestyle="solid", c="green", alpha=0.7, linewidth=0.4)
    # chain04:
    chain_data = data.loc[data["chain"] == 4]
    plt.plot(chain_data.index, chain_data[f"{parameter}"], linestyle="solid", c="cyan", alpha=0.7, linewidth=0.4)
    # chain05:
    chain_data = data.loc[data["chain"] == 5]
    plt.plot(chain_data.index, chain_data[f"{parameter}"], linestyle="solid", c="yellow", alpha=0.7, linewidth=0.4)
    # chain06:
    chain_data = data.loc[data["chain"] == 6]
    plt.plot(chain_data.index, chain_data[f"{parameter}"], linestyle="solid", c="purple", alpha=0.7, linewidth=0.4)

    plt.savefig(f"Caterpillar {parameter}", dpi=300)
    plt.close()
