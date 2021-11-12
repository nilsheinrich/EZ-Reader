import numpy as np
import sys

# works for multiple metrics:


def calc_likelihood(s_star, s, debugme="nope"):
    # calculates a single likelihood from data statistics s
    # and model replicate statistics s_star

    s = np.matrix(s).transpose()
    if debugme == "debug":
        print("s")
        print(s)

    s_star = np.matrix(s_star).transpose()
    # if only a single metric is used to calculate the likelihood do not apply .transpose()
    if debugme == "debug":
        print("s_star")
        print(s_star)

    mu_hat = np.mean(s_star, axis=1)  # axis = 1 for ->
    if debugme == "debug":
        print("mu_hat")
        print(mu_hat)

    S = s_star - mu_hat
    if debugme == "debug":
        print("S")
        print(S)

    Sigma_hat = np.dot(S, np.transpose(S)) / (np.shape(S)[1] - 1)
    if debugme == "debug":
        print("Sigma_hat")
        print(Sigma_hat)

    tmp1 = np.transpose(-1 / 2 * (s - mu_hat))
    if debugme == "debug":
        print("tmp1")
        print(tmp1)

    if np.linalg.cond(Sigma_hat) < (1 / sys.float_info.epsilon):
        tmp2 = np.linalg.inv(Sigma_hat)
    else:
        tmp2 = np.linalg.pinv(Sigma_hat)
    #tmp2 = np.linalg.inv(Sigma_hat)
    if debugme == "debug":
        print("tmp2")
        print(tmp2)

    tmp3 = (s - mu_hat)
    if debugme == "debug":
        print("tmp3")
        print(tmp3)

    tmp4 = -1 / 2 * np.log(np.linalg.norm(Sigma_hat))
    if debugme == "debug":
        print("tmp4")
        print(tmp4)

    l_s = np.dot(np.dot(tmp1, tmp2), tmp3) + tmp4

    if debugme == "debug":
        print("l_s")
        print(l_s)
        print("\n\n")

    return float(l_s)
