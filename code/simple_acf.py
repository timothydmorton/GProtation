from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os

plotpar = {'axes.labelsize': 18,
           'font.size': 10,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def simple_acf(x, y):
    """
    Calculate ACF of y.
    Returns period, acf_smooth, lags, rvar
    """

    # interpolate across gaps
    gap_days = 0.02043365
    time = np.arange(x[0], x[-1], gap_days)
    lin_interp = np.interp(time, x, y)
    x, y = time, lin_interp

    # fit and subtract straight line
    AT = np.vstack((x, np.ones_like(x)))
    ATA = np.dot(AT, AT.T)
    m, b = np.linalg.solve(ATA, np.dot(AT, y))
    y -= m*x + b

    # perform acf
    acf = dan_acf(y)

    # create 'lags' array
    lags = np.arange(len(acf))*gap_days

    N = len(acf)
    double_acf, double_lags = [np.zeros((2*N)) for i in range(2)]
    double_acf[:N], double_lags[:N] = acf[::-1], -lags[::-1]
    double_acf[N:], double_lags[N:] = acf, lags
    acf, lags = double_acf, double_lags

    # smooth with Gaussian kernel convolution
    Gaussian = lambda x, sig: 1./(2*np.pi*sig**.5) * np.exp(-0.5*(x**2)/
                                                            (sig**2))
    conv_func = Gaussian(np.arange(-28, 28, 1.), 9.)
    acf_smooth = np.convolve(acf, conv_func, mode='same')

    # just use the second bit (no reflection)
    acf_smooth, lags = acf_smooth[N:], lags[N:]

    # cut it in half (and reduce to 100 days)
    m = lags < max(lags)/2.
    # m = (lags < max(lags)/2.) * (lags < 100)
    acf_smooth, lags = acf_smooth[m], lags[m]

    # ditch the first point
    acf_smooth, lags = acf_smooth[1:], lags[1:]

    # fit and subtract straight line
    # AT = np.vstack((lags, np.ones_like(lags)))
    # ATA = np.dot(AT, AT.T)
    # m, b = np.linalg.solve(ATA, np.dot(AT, acf_smooth))
    # acf_smooth -= m*lags + b

    # find all the peaks
    peaks = np.array([i for i in range(1, len(lags)-1)
                     if acf_smooth[i-1] < acf_smooth[i] and
                     acf_smooth[i+1] < acf_smooth[i]])

    # find the first and second peaks
    if len(peaks) > 1:
        if acf_smooth[peaks[0]] > acf_smooth[peaks[1]]:
            period = lags[peaks[0]]
        else:
            period = lags[peaks[1]]
    elif len(peaks) == 1:
        period = lags[peaks][0]
    elif not len(peaks):
        period = np.nan

    # find the highest peak
    if len(peaks):
        m = acf_smooth == max(acf_smooth[peaks])
        highest_peak = acf_smooth[m][0]
        period = lags[m][0]
    else:
        period = 0.

    rvar = np.percentile(y, 95)

    return period, acf_smooth, lags, rvar


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


# dan's acf function
def dan_acf(x, axis=0, fast=False):
    """
    Estimate the autocorrelation function of a time series using the FFT.
    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.
    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.
    :param fast: (optional)
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)
    """
    x = np.atleast_1d(x)
    m = [slice(None), ] * len(x.shape)

    # For computational efficiency, crop the chain to the largest power of
    # two if requested.
    if fast:
        n = int(2**np.floor(np.log2(x.shape[axis])))
        m[axis] = slice(0, n)
        x = x
    else:
        n = x.shape[axis]

    # Compute the FFT and then (from that) the auto-correlation function.
    f = np.fft.fft(x-np.mean(x, axis=axis), n=2*n, axis=axis)
    m[axis] = slice(0, n)
    acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[m].real
    m[axis] = 0
    return acf / acf[m]


def nll(pars, x, y):
    model = QP(pars, x)
    return sum(-.5*(y - model)**2)


def QP(pars, x):
    A, l, G, P, m, s, c = pars
    f = A**2 * np.exp(- x**2/(2 * l**2) - G**2 * np.sin(np.pi*x/P)**2) + m*x
    return f / f[0] * s + c


def fit_QP(lags, acf):
    """
    A function for fitting a Quasi-periodic Gaussian process kernel function
    to an ACF.
    The Fourier transform of the ACF is the Power Spectral Density.
    """

    pars_init = [np.exp(1), np.exp(10), np.exp(.1), 19.514, -.07, .5, .55]

    ffti = np.fft.fft(acf)
    fft = (np.real(ffti)**2 + np.imag(ffti)**2)
    rfft = np.fft.rfft(acf)
    plt.clf()
    plt.plot(rfft)
    plt.xlim(0, 100)
    plt.savefig("fft")

    x = np.arange(0, lags[-1], .1)
    model = QP(pars_init, x)
    plt.clf()
    plt.plot(lags, acf, "CornFlowerBlue", lw=2, label="$\mathrm{ACF}$",
             zorder=1)
    plt.plot(x, model, "HotPink", lw=1.5, label="$\mathrm{Model}$",
             zorder=0, ls="--", dashes=(5, 1))
    plt.legend()
    plt.xlabel("$\mathrm{Lags~(Days)}$")
    plt.ylabel("$\mathrm{Autocorrelation}$")
    plt.xlim(0, lags[-1])
    plt.savefig("test")


if __name__ == "__main__":
    DATA_DIR = "simulations/kepler_diffrot_full/"
    FIG_DIR = "/Users/ruthangus/projects/GProtation/documents/figures"
    x, y = np.genfromtxt(os.path.join(DATA_DIR,
                                      "noise_free/lightcurve_0025.txt")).T
                                      # "final/lightcurve_0025.txt")).T
    # period, acf, lags, rvar = simple_acf(x, y)
    # np.savetxt("test_acf.txt", np.vstack((lags, acf)).T)
    lags, acf = np.genfromtxt("test_acf.txt").T
    fit_QP(lags, acf)
