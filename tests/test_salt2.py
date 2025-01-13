import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")

from models.salt2 import getsalt2_spectrum


midfilt = 3
centering = False
realistic = "realistic" 


all_data = np.load(f"../data/goldstein_processed/preprocessed_midfilt_{midfilt}_centering{centering}_{realistic}LSST_phase.npz")
#breakpoint()
training_idx = all_data['training_idx']
testing_idx = all_data['testing_idx']

flux, wavelength, mask = all_data['flux'][testing_idx,:], all_data['wavelength'][testing_idx,:], all_data['mask'][testing_idx,:]
phase = all_data['phase'][testing_idx] 

wavelength_mean, wavelength_std = all_data['wavelength_mean'], all_data['wavelength_std']
phase_mean, phase_std = all_data['phase_mean'], all_data['phase_std']

photoflux, phototime, photomask = all_data['photoflux'][testing_idx,:], all_data['phototime'][testing_idx,:], all_data['photomask'][testing_idx,:]
#phototime = np.concatenate( (phototime, phototime, phototime), 1) # temp fix
photowavelength = all_data['photowavelength'][testing_idx,:]
phototime_mean, phototime_std = all_data['phototime_mean'], all_data['phototime_std']
photoflux_mean, photoflux_std = all_data['photoflux_mean'], all_data['photoflux_std']

#breakpoint()
sed, params = getsalt2_spectrum(phototime[0][photomask[0] == 1] * phototime_std + phototime_mean, 
                           photoflux[0][photomask[0] == 1] * photoflux_std + photoflux_mean, 
                           photowavelength[0][photomask[0] == 1], 
                           wavelength[0] * wavelength_std + wavelength_mean, 
                           phase[0] * phase_std + phase_mean,
                           returnparams=True)

print(params)


