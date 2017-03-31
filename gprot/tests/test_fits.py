import numpy as np
import pandas as pd
import tempfile
import logging
import os, shutil

from gprot.aigrain import AigrainLightCurve
from gprot.kepler import KeplerLightCurve
from gprot.model import GPRotModelCelerite
from gprot.fit import fit_emcee3


def _check_posterior(mod):
    bounds = mod.bounds
    assert np.isfinite(mod.lnpost([(hi+lo)/2. for hi,lo in bounds]))

def _check_fit(mod, nwalkers=20, iter_chunksize=5, maxiter=1, test=True):
    tmp = tempfile.gettempdir()
    chainsdir = os.path.join(tmp, 'mcmc_chains')
    resultsdir = os.path.join(tmp, 'results')
    fit_emcee3(mod, nwalkers=nwalkers, iter_chunksize=iter_chunksize,
                sample_directory=chainsdir, resultsdir=resultsdir,
                maxiter=maxiter)

    samples = pd.read_hdf(os.path.join(resultsdir, '{}.h5'.format(mod.name)), 'samples')
    logging.info(samples.describe())

    shutil.rmtree(chainsdir)
    shutil.rmtree(resultsdir)


def test_kepler(koi=102, sub=5, chunksize=None, acf_prior=True, sap=False):
    kws = dict(sub=sub, chunksize=chunksize, sap=sap, careful_stitching=sap)
    lc = KeplerLightCurve(koi, **kws)  
    mod = GPRotModelCelerite(lc, acf_prior=acf_prior)
    _check_posterior(mod)
    _check_fit(mod)

def test_aigrain(i=6, sub=5, chunksize=None, acf_prior=True):
    kws = dict(sub=sub, chunksize=chunksize)
    lc = AigrainLightCurve(i, **kws)  
    mod = GPRotModelCelerite(lc, acf_prior=acf_prior)
    _check_posterior(mod)
    _check_fit(mod)

