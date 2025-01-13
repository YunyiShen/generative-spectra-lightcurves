import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
import re

midfilt = int(sys.argv[1]) #3
centering = sys.argv[2].lower() == "true" #False
realistic = "realistic" if sys.argv[3].lower() == "true" else "" #"" # "" #if you want high cadency

band_names = ['u', 'g', 'r', 'i', 'z', 'y']
#centering = True
batch_size = 10
num_batches_plot = 15
#realistic = "realistic"
#breakpoint()

fig, axs = plt.subplots(int(np.ceil( batch_size * num_batches_plot/5)), 5+2, figsize=(20, batch_size * num_batches_plot/2 ))
axs = axs.flatten()
i = 0
from tqdm import tqdm
for j in tqdm(range(num_batches_plot)):
    results = np.load(f"../samples/posterior_test_photometrycond_batch{j}_size{batch_size}_Ia_Goldstein_centering{centering}_{realistic}LSST_phase.npz")
    posterior_samples, gt, wavelength, mask = results['posterior_samples'], results['gt'], results['wavelength'], results['mask']
    mask = mask.astype(bool)
    phase = results['phase']
    #photoband = results['photoband']
    phototime, photoflux, photomask = results['phototime'], results['photoflux'], results['photomask']
    identity = results['identity']
    #breakpoint()
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
        axs[i].set_ylim(-2.5, 2.5)
        i += 1
        if i % 7 == 5: # plot light curves
            
            phototimei = phototime[k][photomask[k] == 1]
            photofluxi = photoflux[k][photomask[k] == 1]
            photobandi = np.repeat(np.array([0,1,2,3,4,5]), 10)[photomask[k] == 1]
            for flt in range(6):
                maskflt = photobandi == flt
                axs[i].plot(phototimei[maskflt], 
                            photofluxi[maskflt], 
                            label=f'band {band_names[flt]}')
                axs[i].scatter(phototimei[maskflt],
                            photofluxi[maskflt])
            axs[i].set_title(f'light curve')
            axs[i].legend()
            axs[i].set_ylim((-20.5,-12.5))
            axs[i].invert_yaxis()
            i += 1
        if i % 7 == 6:
            # {kinetic energy [1e51 erg]}_{total mass [msun]}_{mass of nickel+Fe layer [msun]}_{mass of IME layer [msun]}_{mass of CO layer [msun]}_{int(10**m)}.h5
            params = re.findall(r'[-+]?\d*\.\d+e[-+]?\d+', identity[k])
            for jj, num in enumerate(params):
                axs[i].text(0.1, 1 - (jj + 1) * 0.15, num, 
                            fontsize=12, transform=axs[i].transAxes)
            axs[i].set_title('params')
            axs[i].axis('off')
            i += 1

        

plt.tight_layout(rect=[0.04, 0.04, 1, 1])
plt.show()
plt.savefig(f'first_ten_Ia_Goldstein_centered{centering}_{realistic}LSST_phase_more.png')
plt.close()


