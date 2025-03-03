import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from tqdm import tqdm


num_jobs = 10
n_batch = 15
batch_size = 10

salt_test_loss = [[] for i in range(5)]
salt_coverage = [[] for i in range(5)]
salt_width = [[] for i in range(5)]
vdm_test_loss = [[] for i in range(5)]
vdm_coverage = [[] for i in range(5)]
vdm_width = [[] for i in range(5)]
identities = [[] for i in range(5)]


for job in tqdm(range(num_jobs)):
    for batch in range(n_batch):
        results = np.load(f"./samples/posterior_test_photometrycond_job{job}_batch{batch}_size{batch_size}_Ia_Goldstein_centeringFalse_realisticLSST_phase.npz")
        salt3_res = np.load(f"./samples/salt2_samples_job{job}_batch{batch}_size{batch_size}_Ia_Goldstein_centeringFalse_realisticLSST_phase.npz")
        
        gt = results['gt']
        posterior = results['posterior_samples']
        posterior = posterior.reshape(batch_size, 50,posterior.shape[1])

        posterior_mean = np.mean(posterior, axis = 1)
        posterior_upper = np.quantile(posterior, 0.95, axis = 1)
        posterior_lower = np.quantile(posterior, 0.05, axis = 1)

        salt3 = salt3_res['salt_samples']
        salt3 = np.where(salt3 == -np.inf, np.nan, salt3)
        salt3_mean = np.nanmean(salt3, axis = 1)
        salt3_upper = np.nanquantile(salt3, 0.95, axis = 1)
        salt3_lower = np.nanquantile(salt3, 0.05, axis = 1)
        #breakpoint()
        wavelength = results['wavelength']
        mask = results['mask']
        phase = results['phase']
        identity = results['identity']

        for i in range(batch_size):
            phase_i = phase[i]
            which_list = int((phase_i + 10)/10)
            gt_i = gt[i]
            wavelength_i = wavelength[i]
            posterior_mean_i = posterior_mean[i]

            salt3_mean_i = salt3_mean[i]
            salt_test_loss[which_list] += [salt3_mean_i - gt_i]
            vdm_test_loss[which_list] += [posterior_mean_i - gt_i]

            salt_upper_i = salt3_upper[i]
            salt_lower_i = salt3_lower[i]
            posterior_upper_i = posterior_upper[i]
            posterior_lower_i = posterior_lower[i]
            
            cover = np.logical_and(posterior_lower_i < gt_i, gt_i < posterior_upper_i)
            vdm_coverage[which_list] += [cover]
            vdm_width[which_list] += [posterior_upper_i - posterior_lower_i]

            cover = np.logical_and(salt_lower_i < gt_i, gt_i < salt_upper_i)
            salt_coverage[which_list] += [cover]
            salt_width[which_list] += [salt_upper_i - salt_lower_i]

            identities[which_list] += [identity[i]]
        #breakpoint()
        np.savez("./metrics/test_losses.npz",
                salt_test_loss = salt_test_loss,
                salt_coverage = salt_coverage,
                salt_width = salt_width,
                vdm_test_loss = vdm_test_loss,
                vdm_coverage = vdm_coverage,
                vdm_width = vdm_width,
                identities = identities,
                wavelength = wavelength[0,:]
                )


