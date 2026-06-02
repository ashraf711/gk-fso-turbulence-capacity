import numpy as np
import matplotlib.pyplot as plt


def _setup_publication_style():
    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 20,
        "legend.fontsize": 18,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
        "mathtext.fontset": "stix", 
    })



def _finish_plot(outpath, handle_legend=True):
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    
    if handle_legend:
        plt.legend(frameon=False)
        
    plt.tight_layout(pad=0.2)
    plt.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def _gg_label(results):
    if "GG_alpha" in results and "GG_beta" in results:
        a = float(np.asarray(results["GG_alpha"])[0])
        b = float(np.asarray(results["GG_beta"])[0])
        return rf"Gamma-Gamma $(\alpha_g={a:g},\ \beta_g={b:g})$"
    return "Gamma-Gamma"


def plot_capacity_vs_N(results, outpath):
    """
    Capacity comparison versus photon number N.
    No title; suitable for paper captions.
    """
    _setup_publication_style()

    N = np.asarray(results["N"])
    order = np.argsort(N)

 
    plt.figure(figsize=(6.2, 4.4))

    plt.plot(N[order], results["Holevo_Cinf"][order], label=r"Holevo $C_{\infty}$", lw=2.4)
    plt.plot(N[order], results["Dolinar_C1"][order], "--", label=r"Dolinar $C_1$", lw=2.4)

    if "Kennedy_null" in results:
        plt.plot(N[order], results["Kennedy_null"][order], "-.", label=r"Kennedy nulling, $p$ optimized", lw=2.2)
    if "Kennedy_optML" in results:
        plt.plot(N[order], results["Kennedy_optML"][order], ":", label=r"GK ideal, $(\beta,p)$ optimized", lw=2.8)
    if "Kennedy_null_GG" in results:
        plt.plot(N[order], results["Kennedy_null_GG"][order], "-.", label=r"Kennedy nulling, turbulence", lw=2.2)
    if "Kennedy_optML_GG" in results:
        plt.plot(N[order], results["Kennedy_optML_GG"][order], "-", label=rf"GK turbulence, $(\beta,p)$ optimized", lw=2.8)

   
    plt.legend(frameon=False, fontsize=15)
    
    plt.xlabel(r"Mean photon number, $N=|\alpha|^2$")
    plt.ylabel(r"Capacity (bits/symbol)")
    

    _finish_plot(outpath, handle_legend=False)


def plot_optimal_beta_vs_N(results, outpath):
    """
    Separate plot: optimal displacement beta versus photon number N.
    """
    _setup_publication_style()

    N = np.asarray(results["N"])
    order = np.argsort(N)

    plt.figure(figsize=(6.2, 4.4))

    if "beta_star" in results:
        plt.plot(N[order], results["beta_star"][order], label=r"Ideal", lw=2.6)
    if "beta_star_GG" in results:
        plt.plot(N[order], results["beta_star_GG"][order], "--", label=r"Gamma-Gamma turbulence", lw=2.6)

    plt.xlabel(r"Mean photon number, $N=|\alpha|^2$")
    plt.ylabel(r"Optimal displacement, $\beta^\star$")
    _finish_plot(outpath)


def plot_optimal_prior_vs_N(results, outpath):
    """
    Separate plot: optimal prior p versus photon number N.
    """
    _setup_publication_style()

    N = np.asarray(results["N"])
    order = np.argsort(N)

   
    plt.figure(figsize=(6.2, 4.4))

    if "p_star" in results:
        plt.plot(N[order], results["p_star"][order], label=r"Ideal", lw=2.6)
    if "p_star_GG" in results:
        plt.plot(N[order], results["p_star_GG"][order], "--", label=r"Gamma-Gamma turbulence", lw=2.6)

    plt.xlabel(r"Mean photon number, $N=|\alpha|^2$")
    plt.ylabel(r"Optimal prior, $p^\star$")
    plt.ylim(0.0, 1.0)
    _finish_plot(outpath)


def plot_beta_comparison(results, outpath):
    plot_optimal_beta_vs_N(results, outpath)


def plot_prior_comparison(results, outpath):
    plot_optimal_prior_vs_N(results, outpath)