import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from models.data_util import normalizing_spectra
import json

'''
# Plot the first 10 spectra
test_data = np.load("../data/test_data_align_with_simu.npz")
train_data = np.load("../data/train_data_align_with_simu.npz")
fluxes_std,  fluxes_mean = train_data['flux_std'], train_data['flux_mean']
wavelengths_std, wavelengths_mean = train_data['wavelength_std'], train_data['wavelength_mean']

type = train_data['type']
## Ia subtypes in ZTF data: [4,7,8,13,15,17]
Ias = [1,5,11,12,13,19]
keep = np.array([x in Ias for x in train_data['type']])


flux, wavelength, mask = train_data['flux'][keep,:], train_data['wavelength'][keep,:], train_data['mask'][keep,:]


wavelength = wavelength * wavelengths_std + wavelengths_mean


#flux, wavelength = test_data['flux'], test_data['wavelength']
wavelength_cond = (np.linspace(3000., 8000., flux.shape[1])) 
photomask = test_data['photomask']
#mask = test_data['mask']

#breakpoint()
'''
class_encoding = json.load(open('../data/train_class_dict_align_with_simu_centered.json'))
class_decoding = {v:k for k,v in class_encoding.items()}


results = np.load("../samples/posterior_test_photometrycond_first30_memory_Ia_ZTF_centered.npz")
posterior_samples, gt, wavelength, mask = results['posterior_samples'], results['gt'], results['wavelength'], results['mask']
wavelength_cond = (np.linspace(3000., 8000., posterior_samples.shape[1]))
ztfid = results['ztfid'] 
#breakpoint()
#posterior_samples = posterior_samples.reshape(-1, posterior_samples.shape[1])
#posterior_samples = posterior_samples.reshape(gt.shape[0], posterior_samples.shape[1], -1)
#posterior_samples = np.swapaxes(posterior_samples, 0, -1)

types = results['SNtype']
all_type_focus = [0,1,2,3]


posterior_mean = np.mean(posterior_samples, axis=-1)
posterior_lower = np.quantile(posterior_samples, 0.025, axis=-1) 
posterior_upper = np.quantile(posterior_samples, 0.975, axis=-1) 
#breakpoint()
posterior_std = posterior_samples.std(axis=-1)

fig, axs = plt.subplots(5, 6, figsize=(40, 20))
axs = axs.flatten()
i = 0
#breakpoint()
for i in range(30):
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
        running_thingy += [posterior_samples[i * 50 + j , :]]
    running_thingy = np.array(running_thingy)
    #running_thingy -= running_thingy.mean(axis = 1)[:,None]
    posterior_mean = np.median(running_thingy, axis=0)
    posterior_lower = np.quantile(running_thingy, 0.05, axis=0)
    posterior_upper = np.quantile(running_thingy, 0.95, axis=0)
    axs[i].plot(wavelength_cond, posterior_mean, 
                color='blue', label='posterior median')
        #breakpoint()
    axs[i].fill_between(wavelength_cond, 
                                 posterior_lower, 
                                 posterior_upper, 
                        color='blue', alpha=0.3)
    axs[i].legend()
    axs[i].set_ylim(-3, 3)
    axs[i].set_title(f"{ztfid[i]}, {class_decoding[types[i]]}")
plt.tight_layout(rect=[0.04, 0.04, 1, 1])
plt.show()
plt.savefig(f'first_ten_Ia_ZTF_centered.png')
plt.close()

for i in range(30):
    running_thingy = []
    for j in range(50):
        running_thingy += [posterior_samples[i * 50 + j , :]]
        plt.plot(wavelength_cond, running_thingy[-1], 
                color='green', label='posterior sample', alpha=0.01)
    running_thingy = np.array(running_thingy)
    #running_thingy -= running_thingy.mean(axis = 1)[:,None]
    posterior_mean = np.median(running_thingy, axis=0)
    plt.plot(wavelength_cond, posterior_mean, 
                color='blue', label='posterior median')
    
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
plt.ylim(-5, 3)
plt.tight_layout(rect=[0.04, 0.04, 1, 1])
plt.show()
plt.savefig(f'first_ten_Ia_ZTF_together_centered.png')
