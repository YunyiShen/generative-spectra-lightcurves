import h5py
import numpy as np
import astropy
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


def get_photometry_wavelength(filter_code):
    return (filter_code == 1) * 1196.25 + (filter_code == 2) * 6366.38



cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
#distance_modulus = cosmo.distmod(redshift)

# all types ['SLSN-I', 'SN II', 'SN IIn', 'SN Ia', 'SN Ib', 'SN Ic']
file = h5py.File("../data/ZTF_Pretrain_5Class_ZFLAT_PERFECT.hdf5", 'r')
#file['Photometry']['SN Ia']['SNIa-SALT2']
# ['TID', 'filter', 'mag_obs', 'mag_obs_err', 'mag_perfect', 'mjd', 'mjd_trigger', 'mwebv', 'mwebv_err', 'z']

#file['Spectroscopy']['SN Ia']['SNIa-SALT2']
# ['TID', 'flux_obs', 'flux_perfect', 'mjd', 'mjd_trigger', 'mwebv', 'mwebv_err', 'wavelength', 'z']

all_types = ['SLSN-I', 'SN II', 'SN IIn', 'SN Ia', 'SN Ib', 'SN Ic']

spect = []
wavelength = []
spec_time = []
spec_mask = []


photometry = []
photo_wavelength = []
photo_mask = []
photo_time = []

red_shift = []
sn_type = []

photometry_length = 256

for sns in all_types:
    spect_tmp = file['Spectroscopy'][sns]['SNIa-SALT2']['flux_obs']
    spect.append(spect_tmp)

    spec_mask_tmp = spect_tmp != -999.
    spec_mask.append(spec_mask_tmp)

    zs = file['Spectroscopy'][sns]['SNIa-SALT2']['z']
    red_shift.append(zs)

    wavelength_tmp = file['Spectroscopy'][sns]['SNIa-SALT2']['wavelength']/(1 + zs) # redshift correction
    wavelength.append(wavelength_tmp)

    spec_time_tmp = file['Spectroscopy'][sns]['SNIa-SALT2']['mjd'] / (1 + zs)
    spec_time.append(spec_time_tmp)

    photo_mask_tmp = np.array(file['Photometry'][sns]['SNIa-SALT2']['mag_obs']) != -999.
    photo_mask.append(photo_mask_tmp)

    photometry_tmp = np.array(file['Photometry'][sns]['SNIa-SALT2']['mag_obs'])
    photometry_tmp -= cosmo.distmod(np.array(zs)).value[:,None]
    photometry_tmp *= photo_mask_tmp

    photometry.append(photometry_tmp)

    
    

    photo_wavelength_tmp = get_photometry_wavelength(np.array(file['Photometry'][sns]['SNIa-SALT2']['filter']))
    photo_wavelength.append(photo_wavelength_tmp)

    photo_time_tmp = np.array(file['Photometry'][sns]['SNIa-SALT2']['mjd']) / (1 + zs)
    photo_time.append(photo_time_tmp)

    sn_type.append(np.array([sns] * len(photometry_tmp)))


    '''
    TODO: figure out how to get peak and thus phase, 
    what other corrections need to be done, 
    e.g., 
    whether we need to convert magnitudes to fluxes
    whether we need to correct for simulation having longer exposure times
    whether we need to correct for cosmological effects in spectra measurements as in photometry
    
    '''


class_encoding = {cls: idx for idx, cls in enumerate(set(all_types))}
class_list = np.array([class_encoding[cls] for cls in sn_type])