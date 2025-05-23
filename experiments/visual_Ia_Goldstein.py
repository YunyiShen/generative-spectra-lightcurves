import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from models.data_util import normalizing_spectra
import json

centering = True
results = np.load(f"../samples/posterior_test_photometrycond_first10_Ia_Goldstein_centering{centering}_phase.npz")
posterior_samples, gt, wavelength, mask = results['posterior_samples'], results['gt'], results['wavelength'], results['mask']
mask = mask.astype(bool)
#breakpoint()

fig, axs = plt.subplots(2, 5, figsize=(20, 5))
axs = axs.flatten()
i = 0
#breakpoint()
for i in range(10):
    gti = gt[i][mask[i]]
    wavelengthi = wavelength[i][mask[i]]
    in_range = np.where((wavelengthi > 3000) * (wavelengthi < 8000) )
    gti = gti[in_range]
    #gti -= gti.mean()
    #breakpoint()
    wavelengthi = wavelengthi[in_range]
    #gti = (gti-gti.min())/(gti.max()-gti.min())
    axs[i].plot(wavelengthi, 
                gti, color='red', label='ground truth')
        #breakpoint()
    running_thingy = []
    
    for j in range(50):
        running_thingy += [posterior_samples[i * 50 + j , :]]#[posterior_samples[i + j * 10, :]]
    running_thingy = np.array(running_thingy)
    #running_thingy -= running_thingy.mean(axis = 1)[:,None]
    posterior_mean = np.mean(running_thingy, axis=0)
    posterior_lower = np.quantile(running_thingy, 0.05, axis=0)
    posterior_upper = np.quantile(running_thingy, 0.95, axis=0)
    axs[i].plot(wavelength[0,:], posterior_mean, 
                color='blue', label='posterior mean')
        #breakpoint()
    axs[i].fill_between(wavelength[0,:], 
                                 posterior_lower, 
                                 posterior_upper, 
                        color='blue', alpha=0.3)
    axs[i].legend()
    
    #axs[i].set_ylim(-3, 3)
    
plt.tight_layout(rect=[0.04, 0.04, 1, 1])
plt.show()
plt.savefig(f'first_ten_Ia_Goldstein_centered_phase.png')
plt.close()

for i in range(10):
    running_thingy = []
    for j in range(50):
        running_thingy += [posterior_samples[i * 50 + j , :]]#[posterior_samples[i + j * 10, :]]
        plt.plot(wavelength[0,:], running_thingy[-1], 
                color='green', label='posterior sample', alpha=0.01)
    running_thingy = np.array(running_thingy)
    #running_thingy -= running_thingy.mean(axis = 1)[:,None]
    posterior_mean = np.mean(running_thingy, axis=0)
    plt.plot(wavelength[0,:], posterior_mean, 
                color='blue', label='posterior mean')
    
    gti = gt[i][mask[i]]
    #gti -= gti.mean()
    wavelengthi = wavelength[i][mask[i]]
    in_range = np.where((wavelengthi > 3000) * (wavelengthi < 8000) )
    gti = gti[in_range]
    #gti = (gti-gti.min())/(gti.max()-gti.min())
    #breakpoint()
    wavelengthi = wavelengthi[in_range]

    plt.plot(wavelengthi, gti, 
                color='red', label='gt', alpha=0.3)
#plt.legend()
#plt.ylim(-5, 3)
plt.tight_layout(rect=[0.04, 0.04, 1, 1])
plt.show()
plt.savefig(f'first_ten_Ia_Goldstein_together_centered_phase.png')
