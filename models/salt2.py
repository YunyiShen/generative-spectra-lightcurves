import numpy as np
import sncosmo
from sncosmo.salt2utils import SALT2ColorLaw
from astropy.table import Table
import pandas as pd
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

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

    lc = Table()
    lc['time'] = time
    lc['band'] = np.array([bandnames[int(i)] for i in band])
    lc['zpsys'] = np.array(['ab']*len(time))
    lc['zp'] = np.zeros(len(time)) + 25
    lc['flux'] = np.array([ab.band_mag_to_flux(m + 25, b) for m, b in zip(mag, lc['band'])])
    lc['fluxerr'] = 1e-5*np.ones(len(time))
    return lc

def fit_supernova(lc):
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
    bnds = {'t0':(-100,100),'x0':(-1e-3, 1e-3), 'x1':(-3, 3), 'c':(-0.5, 0.5)}
    mod = sncosmo.Model('salt3') #sncosmo.Model(source = source)
    mod.set(z=0.) 
    res = sncosmo.fit_lc(lc, mod, 
                         vparam_names=['t0', 'x0', 'x1', 'c'],
                         bounds=bnds, minsnr=0)
    return res[0].parameters


def _salt2sed(params):
    return params[2]*(m0flux + params[3]*m1flux)*np.exp(-params[4]*cl)



def getsalt2_sed(time, mag, band, returnparams=False):

    lc = makelc(time, mag, band)
    params = fit_supernova(lc)
    breakpoint()
    sed = _salt2sed(params)
    if returnparams:
        return sed, params
    else:
        return sed
