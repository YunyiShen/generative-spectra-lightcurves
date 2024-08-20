import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from models.data_util import normalizing_spectra

# Plot the first 10 spectra
test_data = np.load("../data/testing_simulated_data.npz")
flux, wavelength = test_data['flux'], test_data['wavelength']
wavelengths_mean, wavelengths_std = test_data['wavelength_mean'], test_data['wavelength_std']
wavelength_cond = (np.linspace(3000., 8000., flux.shape[1])) 
photomask = test_data['photomask']
mask = test_data['mask']

#breakpoint()


results = np.load("../samples/posterior_test_photometrycond_first100.npz")
posterior_samples, gt = results['posterior_samples'], results['gt']
types = results['SNtype']
type_focus = 0

posterior_mean = np.median(posterior_samples, axis=-1)
posterior_lower = np.quantile(posterior_samples, 0.05, axis=-1) 
posterior_upper = np.quantile(posterior_samples, 0.95, axis=-1) 
#breakpoint()

posterior_std = posterior_samples.std(axis=-1)

fig, axs = plt.subplots(3, 5, figsize=(25, 10))
axs = axs.flatten()
being_plot = 0
#breakpoint()
for i in range(100):

    if being_plot == 15:
        break
    if photomask[i].sum() < 15 or mask[i].sum() < 20 or types[i] != type_focus:
        continue
    axs[being_plot].plot(wavelength[i][mask[i]] * wavelengths_std + wavelengths_mean, 
                gt[i][mask[i]], color='red', label='ground truth')
    #breakpoint()
    axs[being_plot].plot(wavelength_cond, posterior_mean[i], 
                color='blue', label='posterior median')
    #breakpoint()
    axs[being_plot].fill_between(wavelength_cond, 
                                 posterior_lower[i], 
                                 posterior_upper[i], 
                        color='blue', alpha=0.3)
    axs[being_plot].legend()
    axs[being_plot].set_title(f"Sample {i}")
    being_plot += 1
plt.tight_layout(rect=[0.04, 0.04, 1, 1])
plt.show()
plt.savefig(f'first_ten_type{type_focus}.png', dpi=300)
