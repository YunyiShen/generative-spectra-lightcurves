'''
GP preprocessing of light curves, using tiny Gp (see https://tinygp.readthedocs.io/en/stable/tutorials/quickstart.html)
'''

from tinygp import kernels, GaussianProcess
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as np
import jaxopt


def mean_function(params, X):
    mod = np.exp(-0.5 * np.square((X - params["loc"]) / np.exp(params["log_width"])))
    beta = np.array([1, mod])
    return params["amps"] @ beta



def gp_peak_time(time, flux):
    # normalize time and flux
    time_mean = np.mean(time)
    time_std = np.std(time)
    time = (time - time_mean) / time_std

    time = np.sort(time)
    #breakpoint()
    
    flux_mean = np.mean(flux)
    flux_std = np.std(flux)
    flux = (flux - flux_mean) / flux_std


    def build_gp(params):
        kernel = np.exp(params["log_gp_amp"]) * kernels.Matern52(
            np.exp(params["log_gp_scale"])
        )
        return GaussianProcess(
            kernel,
            time,
            diag=np.exp(params["log_gp_diag"]),
            mean=params['mean'],
        )

    @jax.jit
    def loss(params):
        gp = build_gp(params)
        return -gp.log_probability(flux)


    params = dict(
        log_gp_amp=np.log(0.1),
        log_gp_scale=np.log(3.0),
        log_gp_diag=np.log(0.03),
        mean = 0.,
    )

    solver = jaxopt.ScipyMinimize(fun=loss)
    soln = solver.run(jax.tree_util.tree_map(np.asarray, params))

    time_grid = np.linspace(np.min(time), np.max(time), 500)

    gp = build_gp(soln.params)
    _, cond = gp.condition(flux, time_grid)

    prd = cond.loc

    peak = time_grid[np.argmax(prd)] * time_std + time_mean

    return peak
