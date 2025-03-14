import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from tqdm import tqdm
num_jobs = 10
n_batch = 15
batch_size = 10

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
coverage = [[[] for _ in range(5)] for _ in range(9)]
coverage_salt3 = [[[] for _ in range(5)] for _ in range(9)]
coverage_salt3_recon = [[[] for _ in range(5)] for _ in range(9)]

ii = 0
for alpha in tqdm(alphas):
    for job in tqdm(range(num_jobs)):
        for batch in range(n_batch):
            results = np.load(f"./samples/posterior_test_photometrycond_job{job}_batch{batch}_size{batch_size}_Ia_Goldstein_centeringFalse_realisticLSST_phase.npz")
            salt3_res = np.load(f"./samples/salt2_samples_job{job}_batch{batch}_size{batch_size}_Ia_Goldstein_centeringFalse_realisticLSST_phase.npz")
            salt3_recon = np.load(f"./samples/salt2_spectrareconstruct_job{job}_batch{batch}_size{batch_size}_Ia_Goldstein_centeringFalse_realisticLSST_phase.npz")

            
            gt = results['gt']
            posterior = results['posterior_samples']
            posterior = posterior.reshape(batch_size, 50,posterior.shape[1])
            phase = results['phase']

            posterior_mean = np.mean(posterior, axis = 1)
            posterior_upper = np.quantile(posterior, 1-alpha/2, axis = 1)
            posterior_lower = np.quantile(posterior, alpha/2, axis = 1)
            
            salt3 = salt3_res['salt_samples']
            salt3_mean = np.mean(salt3, axis = 1)
            salt3_upper = np.nanquantile(salt3, 1-alpha/2, axis = 1)
            salt3_lower = np.nanquantile(salt3, alpha/2, axis = 1)

            salt3_recon_samples = salt3_recon['salt_results']
            salt3_recon_upper = np.nanquantile(salt3_recon_samples, 1-alpha/2, axis = 1)
            salt3_recon_lower = np.nanquantile(salt3_recon_samples, alpha/2, axis = 1)


            
            
            for i in range(batch_size):
                phase_i = phase[i]
                which_list = int((phase_i + 10)/10)
                gt_i = gt[i]
                
                posterior_upper_i = posterior_upper[i]
                posterior_lower_i = posterior_lower[i]
                cover = np.logical_and(posterior_lower_i < gt_i, gt_i < posterior_upper_i)
                coverage[ii][which_list] += [cover]

                salt_upper_i = salt3_upper[i]
                salt_lower_i = salt3_lower[i]
                salt_cover = np.logical_and(salt_lower_i < gt_i, gt_i < salt_upper_i)
                coverage_salt3[ii][which_list] += [salt_cover]

                salt_recon_upper_i = salt3_recon_upper[i]
                salt_recon_lower_i = salt3_recon_lower[i]
                salt_cover_recon = np.logical_and(salt_recon_lower_i < gt_i, gt_i < salt_recon_upper_i)
                coverage_salt3_recon[ii][which_list] += [salt_cover_recon]

            #breakpoint()
    ii += 1

np.savez("./metrics/coverage_varing.npz", 
         coverage=coverage,
         coverage_salt3=coverage_salt3,
         coverage_salt3_recon = coverage_salt3_recon,
         alphas=alphas)





