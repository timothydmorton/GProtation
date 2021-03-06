from __future__ import print_function
import numpy as np
from GProt import calc_p_init, mcmc_fit
import pandas as pd
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import h5py

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)

def make_new_df(truths, R_DIR):
    """
    Load all the resulting period measurements and make a new pandas
    dataframe.
    """
    m = truths.DELTA_OMEGA.values == 0

    # get column names
    mfname2 = os.path.join(R_DIR, "0002_mcmc_results.txt")
    apfname2 = os.path.join(R_DIR, "0002_acf_pgram_results.txt")
    mdf2, adf2 = pd.read_csv(mfname2), pd.read_csv(apfname2)

    # assemble master data frame
    mcols, acols = mdf2.columns.values, adf2.columns.values
    mcmc = pd.DataFrame(data=np.zeros((0, len(mcols))), columns=mcols)
    acf_pgram = pd.DataFrame(data=np.zeros((0, len(acols))), columns=acols)
    Ns = []
    for i, id in enumerate(truths.N.values[m]):
        sid = str(int(id)).zfill(4)
        mfname = os.path.join(R_DIR, "{0}_mcmc_results.txt".format(sid))
        afname = os.path.join(R_DIR, "{0}_acf_pgram_results.txt".format(sid))
        if os.path.exists(mfname) and os.path.exists(afname):
            Ns.append(int(sid))
            mcmc = pd.concat([mcmc, pd.read_csv(mfname)], axis=0)
            acf_pgram = pd.concat([acf_pgram, pd.read_csv(afname)], axis=0)

    mcmc["N"], acf_pgram["N"] = np.array(Ns), np.array(Ns)
    truths1 = mcmc.merge(acf_pgram, on="N")
    truths_s = truths.merge(truths1, on="N")
    truths_s.to_csv("truths_extended.csv")
    return truths_s


def plots(truths, DIR):
    """
    Plot the GP and acf results.
    """

    truths_e = make_new_df(truths, DIR)
    m = (truths_e.DELTA_OMEGA.values == 0) \
            * (truths_e.acf_period.values > 0)

    N = truths_e.N.values[m]
    true = truths_e.P_MIN.values[m]
    med = np.exp(truths_e.sigma.values[m])  # period and sigma names swapped
    med_errp = np.exp(truths_e.sigma_errp.values[m])
    med_errm = np.exp(truths_e.sigma_errm.values[m])
    maxlike = np.exp(truths_e.sigma_max.values[m])
    amp = truths_e.AMP.values[m]
    acfs = truths_e.acf_period.values[m]
    acf_errs = truths_e.acf_period_err.values[m]
    gammas = truths_e.gamma_max.values[m]
    ss = truths_e.sigma_max.values[m]
    As = truths_e.A_max.values[m]
    ls = truths_e.l_max.values[m]

    mgp = (np.abs(true - maxlike) / true) < .1
    mgpf = (np.abs(true - maxlike) / true) > .2
    ma = (np.abs(true - acfs) / true) < .1
    maf = (np.abs(true - acfs) / true) > .2

    # plot mcmc results for acf successes
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(np.log(xs), np.log(xs), "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) + 2, "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) - 2, "k--", alpha=.5)
    plt.xlim(0, 4)
    plt.ylim(0, 6)
    plt.title("$\mathrm{MCMC~results~for~ACF~successes}$")
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period})$")
    plt.scatter(np.log(true[ma]), np.log(maxlike[ma]), c=np.log(amp[ma]),
                edgecolor="", cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.savefig(os.path.join(DIR, "mcmc_acf.pdf"))

    # plot mcmc results for acf failures
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(np.log(xs), np.log(xs), "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) + 2, "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) - 2, "k--", alpha=.5)
    plt.xlim(0, 4)
    plt.ylim(0, 6)
    plt.title("$\mathrm{MCMC~results~for~ACF~failures}$")
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period})$")
    plt.scatter(np.log(true[maf]), np.log(maxlike[maf]), c=np.log(amp[maf]),
                edgecolor="", cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.savefig(os.path.join(DIR, "mcmc_acf_fail.pdf"))

    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(np.log(xs), np.log(xs), "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) + 2, "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) - 2, "k--", alpha=.5)
    plt.xlim(0, 4)
    plt.ylim(0, 6)
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period})$")
    plt.title("$\mathrm{ACF~successes}$")
    plt.scatter(np.log(true[ma]), np.log(acfs[ma]), c=np.log(amp[ma]),
                edgecolor="", cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.savefig(os.path.join(DIR, "acf.pdf"))

    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(np.log(xs), np.log(xs), "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) + 2, "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) - 2, "k--", alpha=.5)
    plt.xlim(0, 4)
    plt.ylim(0, 6)
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period})$")
    plt.title("$\mathrm{ACF~failures}$")
    plt.scatter(np.log(true[maf]), np.log(acfs[maf]), c=np.log(amp[maf]),
                edgecolor="", cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.savefig(os.path.join(DIR, "acf_fail.pdf"))

    # plot acf results for mcmc successes
    plt.clf()
    plt.plot(np.log(xs), np.log(xs), "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) + 2, "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) - 2, "k--", alpha=.5)
    plt.xlim(0, 4)
    plt.ylim(0, 6)
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period})$")
    plt.title("$\mathrm{ACF~results~for~MCMC~successes}$")
    plt.scatter(np.log(true[mgp]), np.log(acfs[mgp]), c=np.log(amp[mgp]),
                edgecolor="", cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.savefig(os.path.join(DIR, "acf_mcmc.pdf"))

    # plot acf results for mcmc failures
    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(np.log(xs), np.log(xs), "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) + 2, "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) - 2, "k--", alpha=.5)
    plt.xlim(0, 4)
    plt.ylim(0, 6)
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period})$")
    plt.title("$\mathrm{ACF~results~for~MCMC~failures}$")
    plt.scatter(np.log(true[mgpf]), np.log(acfs[mgpf]), c=np.log(amp[mgpf]),
                edgecolor="", cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.plot(true[N == 44], acfs[N == 44], "r.")
    plt.savefig(os.path.join(DIR, "acf_mcmc_fail.pdf"))

    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(np.log(xs), np.log(xs), "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) + 2, "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) - 2, "k--", alpha=.5)
    plt.xlim(0, 4)
    plt.ylim(0, 6)
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period})$")
    plt.title("$\mathrm{MCMC~successes}$")
    plt.scatter(np.log(true[mgp]), np.log(maxlike[mgp]), c=np.log(amp[mgp]),
                edgecolor="", cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    plt.savefig(os.path.join(DIR, "mcmc.pdf"))

    plt.clf()
    xs = np.arange(0, 100, 1)
    plt.plot(np.log(xs), np.log(xs), "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) + 2./3, "k--", alpha=.5)
    plt.plot(np.log(xs), np.log(xs) - 2./3, "k--", alpha=.5)
    plt.xlim(0, 4)
    plt.ylim(0, 6)
    plt.title("$\mathrm{MCMC~failures}$")
    plt.xlabel("$\ln(\mathrm{Injected~Period})$")
    plt.ylabel("$\ln(\mathrm{Recovered~Period})$")
    plt.scatter(np.log(true[mgpf]), np.log(maxlike[mgpf]), c=np.log(amp[mgpf]),
                edgecolor="", cmap="GnBu",
                vmin=min(np.log(amp)), vmax=max(np.log(amp)), s=50, zorder=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("$\ln\mathrm{(Amplitude)}$")
    m = ((maxlike[mgpf] / true[mgpf]) > 1.3)
    mu = ((maxlike[mgpf] / true[mgpf]) < .5)
    print("Overestimates: ", N[mgpf][m])
    print("Underestimates: ", N[mgpf][mu])
    print("Overestimations")
    for num, over in enumerate(N[mgpf][m]):
        print("ID: ", N[mgpf][m][num], ", true period = ", true[mgpf][m][num],
              ", GP period = ", maxlike[mgpf][m][num], ", ACF period = ",
              acfs[mgpf][m][num])
    print("Underestimations")
    for num, under in enumerate(N[mgpf][mu]):
        print("ID: ", N[mgpf][mu][num], ", true period = ",
              true[mgpf][mu][num], ", GP period = ", maxlike[mgpf][mu][num],
              ", ACF period = ", acfs[mgpf][mu][num])
    num = 0
    plt.plot(np.log(true[mgpf][m][num]), np.log(maxlike[mgpf][m][num]), "r.")
    plt.plot(np.log(true[mgpf][mu][num]), np.log(maxlike[mgpf][mu][num]), "r.")
    plt.savefig(os.path.join(DIR, "mcmc_fail.pdf"))

    plt.clf()
    plt.hist(gammas[mgpf], color="w", alpha=.5, histtype="stepfilled",
            ls="--",
            label="${0:.2} \pm {1:.2}$".format(np.median(gammas[mgpf]),
                                        np.std(gammas[mgpf])))
    plt.hist(gammas[mgp], color="w", alpha=.5, histtype="stepfilled",
            label="${0:.2} \pm {1:.2}$".format(np.median(gammas[mgp]),
                                        np.std(gammas[mgp])))
    plt.legend()
    plt.xlabel("$\ln(\Gamma)$")
    plt.savefig(os.path.join(DIR, "gamma_comp.pdf"))

    plt.clf()
    plt.hist(ls[mgpf], color="w", alpha=.5, histtype="stepfilled",
            ls="--",
            label="${0:.2} \pm {1:.2}$".format(np.median(ls[mgpf]),
                                        np.std(ls[mgpf])))
    plt.hist(ls[mgp], color="w", alpha=.5, histtype="stepfilled",
            label="${0:.2} \pm {1:.2}$".format(np.median(ls[mgp]),
                                        np.std(ls[mgp])))
    plt.legend()
    plt.xlabel("$\ln(l)$")
    plt.savefig(os.path.join(DIR, "l_comp.pdf"))

    plt.clf()
    plt.plot(ls[mgpf], np.log(maxlike[mgpf]), "k.")
    plt.plot(ls[mgp], np.log(maxlike[mgp]), "r.")
    plt.xlabel("$\ln(l)$")
    plt.ylabel("$\ln(period)$")
    plt.savefig(os.path.join(DIR, "l_vs_p.pdf"))

    plt.clf()
    fail_ratio = ls[mgpf]/maxlike[mgpf]
    success_ratio = ls[mgp]/maxlike[mgp]
    plt.hist(fail_ratio, color="w", alpha=.5,
             histtype="stepfilled", ls="--",
             label="${0:.2} \pm {1:.2}$"
             .format(np.median(fail_ratio), np.std(fail_ratio)))
    plt.hist(success_ratio, color="w", alpha=.5, histtype="stepfilled",
             label="${0:.2} \pm {1:.2}$".format(np.median(success_ratio),
                 np.std(success_ratio)))
    plt.legend()
    plt.xlabel("$\ln(l)/\ln(P)$")
    plt.savefig(os.path.join(DIR, "l_vs_p_comp.pdf"))

    plt.clf()
    plt.hist(As[mgpf], color="w", alpha=.5, histtype="stepfilled",
            ls="--",
            label="${0:.2} \pm {1:.2}$".format(np.median(As[mgpf]),
                                        np.std(As[mgpf])))
    plt.hist(As[mgp], color="w", alpha=.5, histtype="stepfilled",
            label="${0:.2} \pm {1:.2}$".format(np.median(As[mgp]),
                                        np.std(As[mgp])))
    plt.legend()
    plt.xlabel("$\ln(A)$")
    plt.savefig(os.path.join(DIR, "A_comp.pdf"))

    plt.clf()
    plt.hist(ss[mgpf], color="w", alpha=.5, histtype="stepfilled",
            ls="--",
            label="${0:.2} \pm {1:.2}$".format(np.median(ss[mgpf]),
                                        np.std(ss[mgpf])))
    plt.hist(ss[mgp], color="w", alpha=.5, histtype="stepfilled",
            label="${0:.2} \pm {1:.2}$".format(np.median(ss[mgp]),
                                        np.std(ss[mgp])))
    plt.legend()
    plt.xlabel("$\ln(\sigma)$")
    plt.savefig(os.path.join(DIR, "s_comp.pdf"))

    plt.clf()
    m = maxlike[mgpf] < 1000
    plt.hist(maxlike[mgpf][m], color="w", alpha=.5, histtype="stepfilled",
            ls="--",
            label="${0:.2} \pm {1:.2}$".format(np.median(maxlike[mgpf][m]),
                                        np.std(maxlike[mgpf][m])))
    plt.hist(maxlike[mgp], color="w", alpha=.5, histtype="stepfilled",
            label="${0:.2} \pm {1:.2}$".format(np.median(maxlike[mgp]),
                                        np.std(maxlike[mgp])))
    plt.legend()
    plt.xlabel("$\ln(\mathrm{Period})$")
    plt.savefig(os.path.join(DIR, "period_comp.pdf"))


if __name__ == "__main__":

    DIR = "../code/simulations/kepler_diffrot_full/par/"
    truths = pd.read_csv(os.path.join(DIR, "final_table.txt"), delimiter=" ")

    print("mcmc Gprior rms = ", plots(truths, "results_sigma"))
