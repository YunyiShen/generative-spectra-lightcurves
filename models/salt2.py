import numpy as np
from sncosmo.salt2utils import SALT2ColorLaw
import sncosmo
from astropy.table import Table
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import pandas as pd

'''
m0phase,m0wavelength,m0flux = np.loadtxt('./salt3_template_0.dat',unpack=True)
m1phase,m1wavelength,m1flux = np.loadtxt('./salt3_template_1.dat',unpack=True)

with open('./salt2_color_correction.dat') as fin:
    lines = fin.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n','')
    colorlaw_salt2_coeffs = np.array(lines[1:5]).astype('float')
    salt2_colormin = float(lines[6].split()[1])
    salt2_colormax = float(lines[7].split()[1])
    colorlaw_salt2 = cl = SALT2ColorLaw([salt2_colormin,salt2_colormax],colorlaw_salt2_coeffs)

spectrum = x0*(m0flux + x1*m1flux)*np.exp(-c*cl)
'''

# register SALT3 model #

source = sncosmo.SALT2Source(modeldir=None,
                             m0file='salt3_template_0.dat',
                             m1file='salt3_template_1.dat',
                             clfile='salt2_color_correction.dat') # this makes it salt3


## register LSST bandpasses ##
bands = 'ugrizy'
for band in bands:
    source = f"../data/filters/LSST/LSST_LSST.{band}.dat"
    band = pd.read_csv(source, header=None, sep='\s+', names=['w','t'])
    bp = sncosmo.Bandpass(band.w.values, band.t.values, name=f'lsst{band}')
    sncosmo.registry.register(bp)

bandnames = ['lsst'+band for band in bands]
ab = sncosmo.get_magsystem('ab')

def makelc(time, mag, band):
    lc = Table()
    lc['time'] = time
    lc['band'] = np.array([bandnames[i] for i in band])
    lc['zpsys'] = np.array(['ab']*len(time))
    lc['zp'] = np.zeros(len(time))
    lc['flux'] = np.array([ab.mag_to_flux(m, b) for m, b in zip(mag, lc['band'])])
    lc['fluxerr'] = 1e-8*np.ones(len(time))
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
    mod = sncosmo.Model('salt2-extended')
    res = sncosmo.fit_lc(lc, mod, 
                         vparam_names=['t0', 'x0', 'x1', 'c'],
                         bounds=bnds, minsnr=0)
    return res[0].parameters