import h5py
import numpy as np
import astropy
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import extinction


def get_photometry_wavelength(filter_code):
    return (filter_code == 1) * 1196.25 + (filter_code == 2) * 6366.38

def normalize_spectrum(spectra):
    return (spectra - np.min(spectra,axis = 1)[:,None]) / (np.max(spectra,axis=1) - np.min(spectra,axis = 1))[:,None]   

def z_score(thingy, subtract_mean = True):
    mean_thingy = np.mean(thingy)
    std_thingy = np.std(thingy)
    return (thingy - (subtract_mean * mean_thingy)) / std_thingy, mean_thingy, std_thingy


cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
#distance_modulus = cosmo.distmod(redshift)

# all types ['SLSN-I', 'SN II', 'SN IIn', 'SN Ia', 'SN Ib', 'SN Ic']
file = h5py.File("../data/ZTF_Pretrain_5Class_ZFLAT_PERFECT.hdf5", 'r')
#file['Photometry']['SN Ia']['SNIa-SALT2']
# ['TID', 'filter', 'mag_obs', 'mag_obs_err', 'mag_perfect', 'mjd', 'mjd_trigger', 'mwebv', 'mwebv_err', 'z']

#file['Spectroscopy']['SN Ia']['SNIa-SALT2']
# ['TID', 'flux_obs', 'flux_perfect', 'mjd', 'mjd_trigger', 'mwebv', 'mwebv_err', 'wavelength', 'z']

#all_types = ['SLSN-I', 'SN II', 'SN IIn', 'SN Ia', 'SN Ib', 'SN Ic']
#all_types = [ 'SN II', 'SN Ia', 'SN Ib', 'SN Ic'] # remove broken classes for now
all_types = [ 'SN Ia'] # remove broken classes for now

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

photometry_len = 60#133
for sns in all_types:
    

    keyy = [key for key in file['Photometry'][sns].keys()][0]
    transient_model = file['Spectroscopy'][sns][keyy]['mwebv']

    spect_tmp = np.array(file['Spectroscopy'][sns][keyy]['flux_perfect'])
    spec_mask_tmp = spect_tmp != -999.
    spec_mask.append(spec_mask_tmp)
    spect_tmp = spect_tmp * spec_mask_tmp
    spect_tmp = normalize_spectrum(spect_tmp)
    spect.append(spect_tmp)
    #breakpoint()
    

    zs = np.array(file['Spectroscopy'][sns][keyy]['z'])
    red_shift.append(zs)

    wavelength_tmp = np.array(file['Spectroscopy'][sns][keyy]['wavelength'])/(1 + zs[:,None]) # redshift correction
    wavelength_tmp *= spec_mask_tmp
    wavelength.append(wavelength_tmp)

    #breakpoint()

    spec_time_tmp = (np.array(file['Spectroscopy'][sns][keyy]['mjd']) - np.array( file['Spectroscopy'][sns][keyy]['mjd_trigger'])) / (1 + zs)
    spec_time.append(spec_time_tmp) # this is phase

    photometry_tmp = np.array(file['Photometry'][sns][keyy]['mag_perfect'])

    if photometry_tmp.shape[1] > photometry_len:
        photometry_tmp = photometry_tmp[:,:photometry_len]
    else:
        photometry_tmp = np.pad(photometry_tmp, ((0,0),(0, photometry_len-photometry_tmp.shape[1])), "constant",constant_values = -999.)
    #breakpoint()
    photo_mask_tmp = np.array(photometry_tmp) != -999.
    photo_mask.append(photo_mask_tmp)

    
    
    # wavelength of photometry
    photo_wavelength_tmp = get_photometry_wavelength(np.array(file['Photometry'][sns][keyy]['filter']))
    if photo_wavelength_tmp.shape[1] > photometry_len:
        photo_wavelength_tmp = photo_wavelength_tmp[:, :photometry_len]
    else:
        photo_wavelength_tmp = np.pad(photo_wavelength_tmp, ((0,0),(0, photometry_len-photo_wavelength_tmp.shape[1])), "constant",constant_values = 0.)
    photo_wavelength.append(photo_wavelength_tmp)




    # bunch of corrections to make for photometry
    photometry_tmp -= cosmo.distmod(np.array(zs)).value[:,None]
    #photometry_tmp = 10 ** (-0.4 * photometry_tmp) # some conversion to flux
    #photometry_tmp[np.where((1 - photo_mask_tmp)==1)] = 0.
    

    #mwebv = np.array(file['Photometry'][sns][keyy]['mwebv'])
    #rv = 3.1
    #breakpoint()
    #extinction_correction = extinction.ccm89(photo_wavelength_tmp, a_v = mwebv*rv, r_v = rv)
    #photometry_tmp -= extinction_correction
    
    photometry_tmp *= photo_mask_tmp
    photometry.append(photometry_tmp)


    photo_time_tmp = (np.array(file['Photometry'][sns][keyy]['mjd']) - np.array(file['Photometry'][sns][keyy]['mjd_trigger'])[:,None]) / (1 + zs[:,None])
    if photo_time_tmp.shape[1] > photometry_len:
        photo_time_tmp = photo_time_tmp[:,:photometry_len]
    else:
        photo_time_tmp = np.pad(photo_time_tmp, ((0,0),(0, photometry_len-photo_time_tmp.shape[1])), "constant",constant_values = 0.)
    photo_time_tmp *= photo_mask_tmp
    


    photo_time.append(photo_time_tmp)

    sn_type.append(np.array([sns] * len(photometry_tmp)))

    #breakpoint()
    '''
    TODO: figure out how to get peak and thus phase: use mjd - mjd_trigger 
    what other corrections need to be done, 
    e.g., 
    whether we need to convert magnitudes to fluxes: NO
    whether we need to correct for simulation having longer exposure times: NO
    whether we need to correct for cosmological effects in spectra measurements as in photometry: NO, we normalize to 0-1
    
    '''

#breakpoint()
class_encoding = {cls: idx for idx, cls in enumerate(set(all_types))}

sn_type = np.concatenate(sn_type, axis = 0)
class_list = np.array([class_encoding[cls] for cls in sn_type])

spect = np.concatenate(spect, axis = 0)
spect, spect_mean, spect_std = z_score(spect)

wavelength = np.concatenate(wavelength, axis = 0)
wavelength, wavelength_mean, wavelength_std = z_score(wavelength)



spec_time = np.concatenate(spec_time, axis = 0)
spec_time, spec_time_mean, spec_time_std = z_score(spec_time)

spec_mask = np.concatenate(spec_mask, axis = 0)

#breakpoint()
photometry = np.concatenate(photometry, axis = 0)
photometry, photometry_mean, photometry_std = z_score(photometry)

photo_wavelength = np.concatenate(photo_wavelength, axis = 0)
photo_wavelength = (photo_wavelength - wavelength_mean)/wavelength_std

photo_mask = np.concatenate(photo_mask, axis = 0)


photo_time = np.concatenate(photo_time, axis = 0)
photo_time, photo_time_mean, photo_time_std = z_score(photo_time,subtract_mean = False)

red_shift = np.concatenate(red_shift, axis = 0)



# random split into training and validation
np.random.seed(42)
idx = np.arange(len(spect))
np.random.shuffle(idx)
split = int(0.85 * len(spect))



np.savez("../data/training_simulated_data_Ia.npz", 
         flux=spect[idx[:split]], 
         wavelength=wavelength[idx[:split]], 
         mask=spec_mask[idx[:split]], 
         type=class_list[idx[:split]], 
         phase=spec_time, photoflux=photometry[idx[:split]], 
         phototime=photo_time[idx[:split]], 
         photowavelength=photo_wavelength[idx[:split]],
         photomask=photo_mask[idx[:split]], 
         
         flux_mean=spect_mean,
         flux_std=spect_std,
         wavelength_mean=wavelength_mean,
         wavelength_std=wavelength_std,
         spectime_mean=spec_time_mean,
         spectime_std=spec_time_std,
         phototime_mean=photo_time_mean,
         phototime_std=photo_time_std,
         photoflux_mean=photometry_mean,
         photoflux_std=photometry_std
        )
import json
with open('../data/train_simulated_class_dict_Ia.json', 'w') as fp:
    json.dump(class_encoding, fp)

np.savez("../data/testing_simulated_data_Ia.npz", 
         flux=spect[idx[split:]], 
         wavelength=wavelength[idx[split:]], 
         mask=spec_mask[idx[split:]], 
         type=class_list[idx[split:]], 
         phase=spec_time, photoflux=photometry[idx[split:]], 
         phototime=photo_time[idx[split:]], 
         photowavelength=photo_wavelength[idx[split:]],
         photomask=photo_mask[idx[split:]], 
         
         flux_mean=spect_mean,
         flux_std=spect_std,
         wavelength_mean=wavelength_mean,
         wavelength_std=wavelength_std,
         spectime_mean=spec_time_mean,
         spectime_std=spec_time_std,
         phototime_mean=photo_time_mean,
         phototime_std=photo_time_std,
         photoflux_mean=photometry_mean,
         photoflux_std=photometry_std
        )
