import numpy as np

from gprot.aigrain import AigrainLightCurve
from gprot.kepler import KeplerLightCurve

from gprot.model import GPRotModelCelerite

def _check_posterior(mod):
    bounds = mod.bounds
    mod.lnpost([(hi+lo)/2. for hi,lo in bounds])

def test_kepler(koi=102, sub=5, chunksize=None, acf_prior=True, sap=False):
    kws = dict(sub=sub, chunksize=chunksize, sap=sap, careful_stitching=sap)
    lc = KeplerLightCurve(koi, **kws)  
    mod = GPRotModelCelerite(lc, acf_prior=acf_prior)
    _check_posterior(mod)

def test_aigrain(i=6, sub=5, chunksize=None, acf_prior=True):
    kws = dict(sub=sub, chunksize=chunksize)
    lc = AigrainLightCurve(i, **kws)  
    mod = GPRotModelCelerite(lc, acf_prior=acf_prior)
    _check_posterior(mod)

