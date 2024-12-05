import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")

centering = True
batch_size = 10
num_batches_plot = 15
#breakpoint()

fig, axs = plt.subplots(int(np.ceil( batch_size * num_batches_plot/5)), 5, figsize=(20, batch_size * num_batches_plot/2 ))
axs = axs.flatten()
i = 0
from tqdm import tqdm
for j in tqdm(range(num_batches_plot)):
    results = np.load(f"../samples/posterior_test_photometrycond_batch{j}_size{batch_size}_Ia_Goldstein_centering{centering}_LSST_phase.npz")
    posterior_samples, gt, wavelength, mask = results['posterior_samples'], results['gt'], results['wavelength'], results['mask']
    mask = mask.astype(bool)
    phase = results['phase']
    for k in range(batch_size):
        gti = gt[k][mask[k]]
        wavelengthi = wavelength[k][mask[k]]
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

        for l in range(50):
            running_thingy += [posterior_samples[k * 50 + l , :]]
        running_thingy = np.array(running_thingy)
        #running_thingy -= running_thingy.mean(axis = 1)[:,None]
        posterior_mean = np.mean(running_thingy, axis=0)
        posterior_lower = np.quantile(running_thingy, 0.025, axis=0)
        posterior_upper = np.quantile(running_thingy, 0.975, axis=0)
        axs[i].plot(wavelength[0,:], posterior_mean, 
                color='blue', label='posterior mean')
        #breakpoint()
        axs[i].fill_between(wavelength[0,:], 
                                 posterior_lower, 
                                 posterior_upper, 
                        color='blue', alpha=0.3)
    
        axs[i].legend()
        axs[i].set_title(f'phase: {phase[k]}')
        i += 1

plt.tight_layout(rect=[0.04, 0.04, 1, 1])
plt.show()
plt.savefig(f'first_ten_Ia_Goldstein_centered_LSST_phase_more.png')
plt.close()

