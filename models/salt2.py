import numpy as np
import sncosmo
from sncosmo.salt2utils import SALT2ColorLaw
from astropy.table import Table
import pandas as pd
import os
from cosmolopy import magnitudes, cc
import math
import astropy.units as u
from astropy.constants import h
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
c = 2.998e10*u.cm/u.s   # Speed of light (cm/s)
c_AAs = (c).to(u.AA/u.s).value
h_erg = h.to(u.erg * u.s).value

# register SALT3 model #
salt3_0 = os.path.join(script_dir, 'salt3_template_0.dat')
salt3_1 = os.path.join(script_dir, 'salt3_template_1.dat')
salt3_color = os.path.join(script_dir, 'salt2_color_correction.dat')


m0phase,m0wavelength,m0flux = np.loadtxt(salt3_0,unpack=True)
m1phase,m1wavelength,m1flux = np.loadtxt(salt3_1,unpack=True)

with open(salt3_color) as fin:
    lines = fin.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n','')
    colorlaw_salt2_coeffs = np.array(lines[1:5]).astype('float')
    salt2_colormin = float(lines[6].split()[1])
    salt2_colormax = float(lines[7].split()[1])
    colorlaw_salt2 = cl = SALT2ColorLaw([salt2_colormin,salt2_colormax],colorlaw_salt2_coeffs)

'''
source = sncosmo.SALT2Source(m0file=salt3_0,
                             m1file=salt3_1,
                             clfile=salt3_color) # this makes it salt3
'''

## register LSST bandpasses ##

bands = 'ugrizy'
bandnames = ['lsst'+band for band in bands]

'''
for thisband in bands:
    source = os.path.join(script_dir, f"../data/filters/LSST/LSST_LSST.{thisband}.dat")
    thisband = pd.read_csv(source, header=None, sep='\s+', names=['w','t'])
    bp = sncosmo.Bandpass(thisband.w.values, thisband.t.values, name=f'mylsst{thisband}')
    breakpoint()
    sncosmo.registry.register(bp)
'''

ab = sncosmo.get_magsystem('ab')

def makelc(time, mag, band):
    zp = 0
    zp_simu = 0#48.6 #48.6
    lc = Table()
    lc['time'] = time
    lc['band'] = np.array([bandnames[int(i)] for i in band])
    lc['zpsys'] = np.array(['ab']*len(time))
    lc['zp'] = np.zeros(len(time)) + zp
    #breakpoint()
    #lc['flux'] = np.array([ab.band_mag_to_flux(m + zp_simu - zp, b) for m, b in zip(mag, lc['band'])])
    lc['flux'] = np.array([ 10**(-0.4 *(m + zp_simu - zp)) for m, b in zip(mag, lc['band'])])
    
    #breakpoint()
    lc['fluxerr'] = 0*lc['flux'] + 0.1 * np.std(lc['flux']) #* lc['flux']/np.max(lc['flux'])
    return lc

def fit_supernova(lc, mcmc = False):
    """
    Small function to fit a light curve with the SALT2 model, using sncosmo and iminuit.
    
    Parameters
    -----------

    lc : astropy.table.Table
        Light curve (in the format sncosmo expects)
    
    Returns
    ----------
    t0, x0, x1, c
        Best-fitting parameters of the model
    """
    #bnds = {'t0':(-10,10),'x0':(-5e-0, 5e-0), 'x1':(-10, 10), 'c':(-1, 1)}
    mod = sncosmo.Model('salt3') #sncosmo.Model(source = source)
    mod.set(z=0.) 
    mod.set(t0=0.)
    bnds = {"t0":(-5,5)}

    if mcmc:
        res = sncosmo.mcmc_lc(lc, mod, 
                             vparam_names=["t0",'x0', 'x1', 'c'],
                             bounds=bnds,
                             minsnr=0, thin=1)
        return res


    res = sncosmo.fit_lc(lc, mod, 
                         vparam_names=["t0",'x0', 'x1', 'c'],
                         bounds=bnds,#None,#bnds, 
                         minsnr=0)
    return res[0].parameters


def fit_supernova_spectra(spectra, mcmc = False):
    """
    Small function to fit a light curve with the SALT2 model, using sncosmo and iminuit.
    
    Parameters
    -----------

    lc : astropy.table.Table
        Light curve (in the format sncosmo expects)
    
    Returns
    ----------
    t0, x0, x1, c
        Best-fitting parameters of the model
    """
    #bnds = {'t0':(-10,10),'x0':(-5e-0, 5e-0), 'x1':(-10, 10), 'c':(-1, 1)}
    bnds = {'t0':(-20,20)}
    mod = sncosmo.Model('salt3') #sncosmo.Model(source = source)
    mod.set(z=0.) 
    mod.set(t0=0.)
    #bnds = {"t0":(-15,15)}

    


    res = sncosmo.fit_lc(model = mod, spectra = spectra,  
                         vparam_names=['t0','x0', 'x1', 'c'],
                         bounds=bnds,#None,#bnds, 
                         minsnr=0)
    
    if mcmc:
        params = res[0].parameters
        mod.set(z=0., t0=params[1], x0=params[2], x1=params[3], c=params[4])
        res = mcmc_spectra(spectra = spectra, model = mod,  
                             vparam_names=["t0",'x0', 'x1', 'c'],
                             bounds=bnds,
                             minsnr=0, thin=1)
        #breakpoint()
        return res

    return res[0].parameters


#def _salt2sed(params):
#    return params[2]*(m0flux + params[3]*m1flux)*np.exp(-params[4]*cl)



def getsalt2_spectrum(time, mag, band, wavelength, phase, returnparams=False):

    lc = makelc(time, mag, band)
    params = fit_supernova(lc)
    mod = sncosmo.Model('salt3')
    mod.set(z=params[0], t0=params[1], x0=params[2], x1=params[3], c=params[4])
    
    #figg = sncosmo.plot_lc(lc, model=mod)
    #figg.savefig('salt2fit.png')
    #plt.close()
    #breakpoint()
    

    spec = np.log10(mod.flux(phase, wavelength.astype(float)) * wavelength ** 2 / c_AAs ) #/ 1e-8
    #breakpoint()
    #sed = _salt2sed(params)
    if returnparams:
        return spec, params
    else:
        return spec


def getsalt2_spectra_samples(time, mag, band, wavelength, phase):

    lc = makelc(time, mag, band)
    res = fit_supernova(lc, mcmc=True)
    mod = sncosmo.Model('salt3')
    samples = res[0].samples
    N = samples.shape[0]
    # random subsample to 200
    #breakpoint()
    n_wanted = 500
    if N > n_wanted:
        indices = np.linspace(1000, N - 1, n_wanted, dtype=int)
        #breakpoint()
        samples = samples[indices]
        N = n_wanted
    
    spectra = []
    
    for i in range(N):
        params = samples[i]
        #breakpoint()
        mod.set(z=0., t0=params[0], x0=params[1], x1=params[2], c=params[3])    
        spec = np.log10(mod.flux(phase, wavelength.astype(float)) * wavelength ** 2 / c_AAs ) #/ 1e-8
        spectra.append(spec)
    
    return np.array(spectra)


def salt2_spectra_reconstruction(wavelengths, 
                                 logfluxs, 
                                 phases, 
                                 returnparams=False):
    spectra = []
    #fluxese = 10**np.array(logfluxs)
    fluxese = (10**np.array(logfluxs)) * c_AAs / np.array(wavelengths)**2
    std = np.std(fluxese)
    for wavelength, logflux, phase in zip(wavelengths, logfluxs, phases):
        ordering = np.argsort(wavelength)
        flux = 10**logflux[ordering]
        wavelength = wavelength[ordering]
        flux = flux * c_AAs / wavelength**2
        spectrum = sncosmo.Spectrum(wavelength, flux, 
                               0*flux + 0.001 * std, 
                               time=phase)
        spectra.append(spectrum)
    
    params = fit_supernova_spectra(spectra, mcmc=False)
    #breakpoint()
    mod = sncosmo.Model('salt3')
    mod.set(z=params[0], t0=params[1], x0=params[2], x1=params[3], c=params[4])
    
    #figg = sncosmo.plot_lc(lc, model=mod)
    #figg.savefig('salt2fit.png')
    #plt.close()
    #breakpoint()
    
    specs = []
    for wavelength, phase in zip(wavelengths, phases):
        spec = np.log10(mod.flux(phase, wavelength.astype(float)) * wavelength ** 2 / c_AAs ) #/ 1e-8
        specs.append(spec)
    #breakpoint()
    #sed = _salt2sed(params)
    if returnparams:
        return specs, params
    else:
        return specs


def salt2_spectra_reconstruction_mcmc(wavelengths, 
                                 logfluxs, 
                                 phases):
    spectra = []
    fluxese = (10**np.array(logfluxs)) * c_AAs / (np.array(wavelengths)**2)
    std = np.std(fluxese)
    for wavelength, logflux, phase in zip(wavelengths, logfluxs, phases):
        ordering = np.argsort(wavelength)
        flux = 10**logflux[ordering]
        wavelength = wavelength[ordering]
        flux = flux * c_AAs / wavelength**2
        spectrum = sncosmo.Spectrum(wavelength, flux, 
                               0*flux + 0.1 * std, 
                               time=phase)
        spectra.append(spectrum)
    
    samples = fit_supernova_spectra(spectra, mcmc=True)[0].samples
    #breakpoint()
    N = samples.shape[0]
    # random subsample to 200
    #breakpoint()
    n_wanted = 500
    if N > n_wanted:
        indices = np.linspace(1000, N - 1, n_wanted, dtype=int)
        #breakpoint()
        samples = samples[indices]
        N = n_wanted
    
    spectra = [[] for i in range(5)]
    
    for i in range(N):
        params = samples[i]
        #breakpoint()
        mod = sncosmo.Model('salt3')
        mod.set(z=0, t0=params[0], x0=params[1], x1=params[2], c=params[3])
        phase_i = 0
        for wavelength, phase in zip(wavelengths, phases):
            spec = np.log10(mod.flux(phase, wavelength.astype(float)) * wavelength ** 2 / c_AAs )
            spectra[phase_i].append(spec)
            phase_i += 1
    
    for i in range(len(spectra)):
        spectra[i] = np.array(spectra[i])
    
    #figg = sncosmo.plot_lc(lc, model=mod)
    #figg.savefig('salt2fit.png')
    #plt.close()
    #breakpoint()

    return spectra
    


def salt2_spectra_reconstruction_cv(wavelengths, 
                                 logfluxs, 
                                 phases, 
                                 returnparams=False):
    '''
    cross validation type of reconstruction, we will keep one out and reconstruct it
    '''
    n_spectra = len(wavelengths)
    
    fluxese = 10**np.array(logfluxs)
    std = np.std(fluxese)
    specs = []
    for i in range(n_spectra):
        spectra = []
        kk = 0
        # leave one out
        for wavelength, logflux, phase in zip(wavelengths, logfluxs, phases):
            
            
            ordering = np.argsort(wavelength)
            
            wavelength = wavelength[ordering]
            
            if kk == i:
                phase_query = phase
                wavelength_query = wavelength
                kk += 1
                continue
            flux = 10**logflux[ordering]
            flux = flux * c_AAs / wavelength**2
            spectrum = sncosmo.Spectrum(wavelength, flux, 
                               0*flux + 0.01 * std, 
                               time=phase)
            spectra.append(spectrum)
            kk += 1
        #breakpoint()
        params = fit_supernova_spectra(spectra, mcmc=False)
        #breakpoint()
        mod = sncosmo.Model('salt3')
        mod.set(z=params[0], t0=params[1], x0=params[2], x1=params[3], c=params[4])
    
        spec = np.log10(mod.flux(phase_query, wavelength_query.astype(float)) * wavelength_query ** 2 / c_AAs ) #/ 1e-8
        specs.append(spec)
    #breakpoint()
    #sed = _salt2sed(params)
    if returnparams:
        return specs, params
    else:
        return specs

from collections import OrderedDict
from sncosmo.fitting import cut_bands, t0_bounds, guess_t0_and_amplitude, generate_chisq
from sncosmo.utils import Result
import copy

def mcmc_spectra(spectra, model, vparam_names, bounds=None, priors=None,
            guess_amplitude=False, guess_t0=False, guess_z=False,
            minsnr=5., modelcov=False, nwalkers=10, nburn=200,
            nsamples=1000, sampler='ensemble', thin=1, a=2.0,
            warn=True):
    """Run an MCMC chain to get model parameter samples.

    This is a convenience function around `emcee.EnsembleSampler`.
    It defines the likelihood function and makes a heuristic guess
    at a good set of starting points for the walkers. It then runs
    the sampler, starting with a burn-in run.

    If you're not getting good results, you might want to try
    increasing the burn-in, increasing the walkers, or specifying a
    better starting position.  To get a better starting position, you
    could first run `~sncosmo.fit_lc`, then run this function with all
    ``guess_[name]`` keyword arguments set to False, so that the
    current model parameters are used as the starting point.

    Parameters
    ----------
    data : `~astropy.table.Table` or `~numpy.ndarray` or `dict`
        Table of photometric data. Must include certain columns.
        See the "Photometric Data" section of the documentation for
        required columns.
    model : `~sncosmo.Model`
        The model to fit.
    vparam_names : iterable
        Model parameters to vary.
    bounds : `dict`, optional
        Bounded range for each parameter. Keys should be parameter
        names, values are tuples. If a bound is not given for some
        parameter, the parameter is unbounded. The exception is
        ``t0``: by default, the minimum bound is such that the latest
        phase of the model lines up with the earliest data point and
        the maximum bound is such that the earliest phase of the model
        lines up with the latest data point.
    priors : `dict`, optional
        Prior probability functions. Keys are parameter names, values are
        functions that return probability given the parameter value.
        The default prior is a flat distribution.
    guess_amplitude : bool, optional
        Whether or not to guess the amplitude from the data. If false, the
        current model amplitude is taken as the initial value. Only has an
        effect when fitting amplitude. Default is True.
    guess_t0 : bool, optional
        Whether or not to guess t0. Only has an effect when fitting t0.
        Default is True.
    guess_z : bool, optional
        Whether or not to guess z (redshift). Only has an effect when fitting
        redshift. Default is True.
    minsnr : float, optional
        When guessing amplitude and t0, only use data with signal-to-noise
        ratio (flux / fluxerr) greater than this value. Default is 5.
    modelcov : bool, optional
        Include model covariance when calculating chisq. Default is False.
    nwalkers : int, optional
        Number of walkers in the sampler.
    nburn : int, optional
        Number of samples in burn-in phase.
    nsamples : int, optional
        Number of samples in production run.
    sampler: str, optional
        The kind of sampler to use. Currently only 'ensemble' for
        `emcee.EnsembleSampler` is supported.
    thin : int, optional
        Factor by which to thin samples in production run. Output samples
        array will have (nsamples/thin) samples.
    a : float, optional
        Proposal scale parameter passed to the sampler.
    warn : bool, optional
        Issue a warning when dropping bands outside the wavelength range of
        the model. Default is True.

        *New in version 1.5.0*

    Returns
    -------
    res : Result
        Has the following attributes:

        * ``param_names``: All parameter names of model, including fixed.
        * ``parameters``: Model parameters, with varied parameters set to
          mean value in samples.
        * ``vparam_names``: Names of parameters varied. Order of parameters
          matches order of samples.
        * ``samples``: 2-d array with shape ``(N, len(vparam_names))``.
          Order of parameters in each row  matches order in
          ``res.vparam_names``.
        * ``covariance``: 2-d array giving covariance, measured from samples.
          Order corresponds to ``res.vparam_names``.
        * ``errors``: dictionary giving square root of diagonal of covariance
          matrix for varied parameters. Useful for ``plot_lc``.
        * ``mean_acceptance_fraction``: mean acceptance fraction for all
          walkers in the sampler.
        * ``ndof``: Number of degrees of freedom (len(data) -
          len(vparam_names)).
          *New in version 1.5.0.*
        * ``data_mask``: Boolean array the same length as data specifying
          whether each observation was used.
          *New in version 1.5.0.*

    est_model : `~sncosmo.Model`
        Copy of input model with varied parameters set to mean value in
        samples.

    """

    try:
        import emcee
    except ImportError:
        raise ImportError("mcmc_lc() requires the emcee package.")
    '''
    # Standardize and normalize data.
    data = photometric_data(data)

    # sort by time
    
    if not np.all(np.ediff1d(data.time) >= 0.0):
        sortidx = np.argsort(data.time)
        data = data[sortidx]
    else:
        sortidx = None
    '''

    # Make a copy of the model so we can modify it with impunity.
    model = copy.copy(model)

    bounds = copy.deepcopy(bounds) if bounds else {}
    if priors is None:
        priors = {}

    # Check that vparam_names isn't empty, check for unknown parameters.
    if len(vparam_names) == 0:
        raise ValueError("no parameters supplied")
    for names in (vparam_names, bounds, priors):
        for name in names:
            if name not in model.param_names:
                raise ValueError("Parameter not in model: " + repr(name))

    # Order vparam_names the same way it is ordered in the model:
    vparam_names = [s for s in model.param_names if s in vparam_names]
    ndim = len(vparam_names)

    # Check that 'z' is bounded (if it is going to be fit).
    if 'z' in vparam_names:
        if 'z' not in bounds or None in bounds['z']:
            raise ValueError('z must be bounded if allowed to vary.')
        if guess_z:
            model.set(z=sum(bounds['z']) / 2.)
        if model.get('z') < bounds['z'][0] or model.get('z') > bounds['z'][1]:
            raise ValueError('z out of range.')

    # Cut bands that are not allowed by the wavelength range of the model.
    '''
    fitdata, data_mask = cut_bands(data, model,
                                   z_bounds=bounds.get('z', None),
                                   warn=warn)
    '''

    # Find t0 bounds to use, if not explicitly given
    '''
    if 't0' in vparam_names and 't0' not in bounds:
        bounds['t0'] = t0_bounds(fitdata, model)'
    '''

    # Note that in the parameter guessing below, we assume that the source
    # amplitude is the 3rd parameter of the Model (1st parameter of the Source)

    # Turn off guessing if we're not fitting the parameter.
    if model.param_names[2] not in vparam_names:
        guess_amplitude = False
    if 't0' not in vparam_names:
        guess_t0 = False

    # Make guesses for t0 and amplitude.
    # (we assume amplitude is the 3rd parameter of the model.)
    '''
    if guess_amplitude or guess_t0:
        t0, amplitude = guess_t0_and_amplitude(fitdata, model, minsnr)
        if guess_amplitude:
            model.parameters[2] = amplitude
        if guess_t0:
            model.set(t0=t0)
    '''

    # Indicies used in probability function.
    # modelidx: Indicies of model parameters corresponding to vparam_names.
    # idxbounds: tuples of (varied parameter index, low bound, high bound).
    # idxpriors: tuples of (varied parameter index, function).
    modelidx = np.array([model.param_names.index(k) for k in vparam_names])
    idxbounds = [(vparam_names.index(k), bounds[k][0], bounds[k][1])
                 for k in bounds]
    idxpriors = [(vparam_names.index(k), priors[k]) for k in priors]

    # Posterior function.
    #chisqrfun = generate_chisq(None, model, spectra,modelcov=modelcov)
    spectra = np.atleast_1d(spectra)
    spectra_invcovs = []
    for spectrum in spectra:
        spectra_invcovs.append(np.linalg.pinv(spectrum.fluxcov))
        #breakpoint()
    def lnlike(parameters):
        for i, low, high in idxbounds:
            if not low < parameters[i] < high:
                return -np.inf

        model.parameters[modelidx] = parameters

        full_chisq = 0.
        for spectrum, spec_invcov in zip(spectra, spectra_invcovs):
            sample_wave, sampling_matrix = \
                    spectrum.get_sampling_matrix()
            sample_flux = model.flux(spectrum.time, sample_wave)
            spec_model_flux = (
                    sampling_matrix.dot(sample_flux) /
                    sampling_matrix.dot(np.ones_like(sample_flux))
                )
            spec_diff = spectrum.flux - spec_model_flux
            spec_chisq = spec_invcov.dot(spec_diff).dot(spec_diff)

            full_chisq += spec_chisq

            #breakpoint()

        logp = -0.5 * full_chisq
        return logp

    def lnprior(parameters):
        logp = 0
        for i, func in idxpriors:
            logp += math.log(func(parameters[i]))
        return logp

    def lnprob(parameters):
        return lnprior(parameters) + lnlike(parameters)

    # Moves to use
    moves = emcee.moves.StretchMove(a=a)

    if sampler == 'ensemble':
        # Heuristic determination of walker initial positions: distribute
        # walkers in a symmetric gaussian ball, with heuristically
        # determined scale.
        ctr = model.parameters[modelidx]
        scale = np.ones(ndim)
        for i, name in enumerate(vparam_names):
            if name in bounds:
                scale[i] = 0.0001 * (bounds[name][1] - bounds[name][0])
            elif model.get(name) != 0.:
                scale[i] = 0.01 * model.get(name)
            else:
                scale[i] = 0.1
        pos = ctr + scale * np.random.normal(size=(nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, moves=moves)

    else:
        raise ValueError('Invalid sampler type. Currently only '
                         '"ensemble" is supported.')

    # Run the sampler.
    pos, prob, state = sampler.run_mcmc(pos, nburn)  # burn-in
    sampler.reset()
    sampler.run_mcmc(pos, nsamples, thin_by=thin)  # production run
    samples = sampler.get_chain(flat=True).reshape(-1, ndim)

    # Summary statistics.
    vparameters = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=0)
    model.set(**dict(zip(vparam_names, vparameters)))
    errors = OrderedDict(zip(vparam_names, np.sqrt(np.diagonal(cov))))
    mean_acceptance_fraction = np.mean(sampler.acceptance_fraction)

    # If we need to, unsort the mask so mask applies to input data
    '''
    if sortidx is not None:
        unsort_idx = np.argsort(sortidx)  # indicies that will unsort array
        data_mask = data_mask[unsort_idx]
    '''
    res = Result(param_names=copy.copy(model.param_names),
                 parameters=model.parameters.copy(),
                 vparam_names=vparam_names,
                 samples=samples,
                 covariance=cov,
                 errors=errors,
                 ndof= None, #len(fitdata) - len(vparam_names),
                 mean_acceptance_fraction=mean_acceptance_fraction,
                 data_mask = None#data_mask
                 )

    return res, model