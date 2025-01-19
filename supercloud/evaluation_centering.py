import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from tqdm import tqdm

from models.salt2 import getsalt2_spectrum

num_jobs = 10
n_batch = 15
batch_size = 10

salt_test_loss = [[] for i in range(5)]
vdm_test_loss = [[] for i in range(5)]
vdm_coverage = [[] for i in range(5)]
vdm_width = [[] for i in range(5)]
identities = [[] for i in range(5)]


for job in tqdm(range(num_jobs)):
    for batch in range(n_batch):
        results = np.load(f"./samples/posterior_test_photometrycond_job{job}_batch{batch}_size{batch_size}_Ia_Goldstein_centeringFalse_realisticLSST_phase.npz")
        gt = results['gt']
        posterior = results['posterior_samples']
        posterior = posterior.reshape(batch_size, 50,posterior.shape[1])
        posterior = posterior - np.mean(posterior, axis = 2)[:,:,None]

        posterior_mean = np.mean(posterior, axis = 1)
        posterior_upper = np.quantile(posterior, 0.95, axis = 1)
        posterior_lower = np.quantile(posterior, 0.05, axis = 1)

        wavelength = results['wavelength']
        mask = results['mask']

        # useful for salt2
        phase = results['phase']
        photoflux = results['photoflux']
        phototime = results['phototime']
        photomask = results['photomask']
        photoband = results['photoband']
        identity = results['identity']
        for i in range(batch_size):
            phase_i = phase[i]
            which_list = int((phase_i + 10)/10)
            gt_i = gt[i] - np.mean(gt[i])
            wavelength_i = wavelength[i]
            posterior_mean_i = posterior_mean[i]
            vdm_test_loss[which_list] += [posterior_mean_i - gt_i]
            posterior_upper_i = posterior_upper[i]
            posterior_lower_i = posterior_lower[i]
            cover = np.logical_and(posterior_lower_i < gt_i, gt_i < posterior_upper_i)
            vdm_coverage[which_list] += [cover]
            vdm_width[which_list] += [posterior_upper_i - posterior_lower_i]

            try:
                salt2_res,param = getsalt2_spectrum(
                    phototime[i][photomask[i] == 1], 
                    photoflux[i][photomask[i] == 1], 
                    photoband[i][photomask[i] == 1], 
                    wavelength_i, 
                    phase_i,
                    returnparams=True)
            except:
                salt2_res = gt_i + np.nan
            
            salt2_res = salt2_res - np.nanmean(salt2_res)
            
            salt_test_loss[which_list] += [salt2_res - gt_i]
            identities[which_list] += [identity[i]]
        np.savez("./metrics/test_losses_centering.npz",
                salt_test_loss = salt_test_loss,
                vdm_test_loss = vdm_test_loss,
                vdm_coverage = vdm_coverage,
                vdm_width = vdm_width,
                identities = identities,
                wavelength = wavelength[0,:]
                )


