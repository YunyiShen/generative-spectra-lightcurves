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
flux_mean, flux_std = all_data['flux_mean'], all_data['flux_std']

photoflux, phototime, photomask = all_data['photoflux'][testing_idx,:], all_data['phototime'][testing_idx,:], all_data['photomask'][testing_idx,:]
#phototime = np.concatenate( (phototime, phototime, phototime), 1) # temp fix
photowavelength = all_data['photowavelength'][testing_idx,:]
phototime_mean, phototime_std = all_data['phototime_mean'], all_data['phototime_std']
photoflux_mean, photoflux_std = all_data['photoflux_mean'], all_data['photoflux_std']

#breakpoint()
which = 38 #38 #106#36
this_phototime = phototime[which][photomask[which] == 1] * phototime_std + phototime_mean
this_photoflux = photoflux[which][photomask[which] == 1] * photoflux_std + photoflux_mean
this_photowavelength = photowavelength[which][photomask[which] == 1]
actual_flux = flux[which] * flux_std + flux_mean


bands = 'ugrizy'
bandnames = ['lsst'+band for band in bands]
for i in range(6):
    plt.plot(this_phototime[this_photowavelength == i], 
             this_photoflux[this_photowavelength == i], label=f"{bandnames[i]}")
    plt.scatter(this_phototime[this_photowavelength == i], 
             this_photoflux[this_photowavelength == i])
# invert y
plt.ylim((-20.5,-12.5))
plt.gca().invert_yaxis()
plt.legend()
plt.show()
plt.savefig("test_lc.png")
plt.close()


sed, params = getsalt2_spectrum(this_phototime, this_photoflux, 
                                this_photowavelength,
                           wavelength[which] * wavelength_std + wavelength_mean, 
                           phase[which] * phase_std + phase_mean,
                           returnparams=True)
plt.plot(wavelength[which] * wavelength_std + wavelength_mean, 
         actual_flux, label="Actual")
plt.plot(wavelength[which] * wavelength_std + wavelength_mean,
            sed, label="SALT2")
plt.legend()
plt.show()
plt.savefig("test_salt2pred.png")

print(params)


