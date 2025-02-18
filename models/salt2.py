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

    if mcmc:
        res = sncosmo.mcmc_lc(model = mod, spectra = spectra, 
                             vparam_names=["t0",'x0', 'x1', 'c'],
                             bounds=bnds,
                             minsnr=0, thin=1)
        return res


    res = sncosmo.fit_lc(model = mod, spectra = spectra,  
                         vparam_names=['t0','x0', 'x1', 'c'],
                         bounds=bnds,#None,#bnds, 
                         minsnr=0)
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
    fluxese = 10**np.array(logfluxs)
    std = np.std(fluxese)
    for wavelength, logflux, phase in zip(wavelengths, logfluxs, phases):
        ordering = np.argsort(wavelength)
        flux = 10**logflux[ordering]
        wavelength = wavelength[ordering]
        flux = flux * c_AAs / wavelength**2
        spectrum = sncosmo.Spectrum(wavelength, flux, 
                               0*flux + 0.01 * std, 
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
