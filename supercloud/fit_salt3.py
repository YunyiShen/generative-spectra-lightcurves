import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from tqdm import tqdm

from models.salt2 import getsalt2_spectra_samples

num_jobs = 60
n_batch = 15
batch_size = 10


def main():
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    #all_starting_points = [i for i in range(num_tasks)]
    #breakpoint()
    starting_points = range(num_jobs)[ my_task_id:num_jobs:num_tasks]

    for job in starting_points:
        for batch in tqdm(range(n_batch)):
            salt_samples = []
            results = np.load(f"./samples/posterior_test_photometrycond_job{job}_batch{batch}_size{batch_size}_Ia_Goldstein_centeringFalse_realisticLSST_phase.npz")
            gt = results['gt']
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
                gt_i = gt[i]
                wavelength_i = wavelength[i]

                try:
                    salt2_res = getsalt2_spectra_samples(
                        phototime[i][photomask[i] == 1], 
                        photoflux[i][photomask[i] == 1], 
                        photoband[i][photomask[i] == 1], 
                        wavelength_i, 
                        phase_i)
                except:
                    salt2_res = (gt_i + np.nan).repeat(200, axis = 0)
                salt_samples.append(salt2_res)
            salt_samples = np.array(salt_samples)
            np.savez(f"./samples/salt2_samples_job{job}_batch{batch}_size{batch_size}_Ia_Goldstein_centeringFalse_realisticLSST_phase.npz", 
                salt_samples=salt_samples,
                gt=gt,
                wavelength=wavelength,
                mask=mask,
                phase=phase,
                photoflux=photoflux,
                phototime=phototime,
                photomask=photomask,
                photoband=photoband,
                identity=identity
                )


if __name__ == '__main__':
    main()

