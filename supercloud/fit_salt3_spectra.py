import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from tqdm import tqdm

from models.salt2 import salt2_spectra_reconstruction_mcmc

num_jobs = 60
n_batch = 15
batch_size = 10
num_phases = 5
def main():
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    #all_starting_points = [i for i in range(num_tasks)]
    #breakpoint()
    starting_points = range(num_jobs)[ my_task_id:num_jobs:num_tasks]

    for job in starting_points:
        for batch in tqdm(range(n_batch)):
            salt_results = []
            results = np.load(f"./samples/posterior_test_photometrycond_job{job}_batch{batch}_size{batch_size}_Ia_Goldstein_centeringFalse_realisticLSST_phase.npz")
            gt = results['gt']
            wavelength = results['wavelength']
            mask = results['mask']
        
            # useful for salt2
            phase = results['phase']
            identity = results['identity']
            #breakpoint()
            for start_at in range(0, batch_size, num_phases):
                fluxes = []
                wavelengths = []
                phases = []
                for offset in range(num_phases):
                    i = start_at + offset
                    fluxes.append(gt[i])
                    wavelengths.append(wavelength[i])
                    phases.append(phase[i])

                try:
                    salt2_res = salt2_spectra_reconstruction_mcmc(wavelengths, 
                           fluxes, 
                           phases)
                except:
                    salt2_res = [(fluxes[0] + np.nan) for _ in range(num_phases)]
                #breakpoint()
                salt_results += salt2_res
            salt_results = np.array(salt_results)
            #breakpoint()
            np.savez(f"./samples/salt2_spectrareconstruct_job{job}_batch{batch}_size{batch_size}_Ia_Goldstein_centeringFalse_realisticLSST_phase.npz", 
                salt_results=salt_results,
                gt=gt,
                wavelength=wavelength,
                mask=mask,
                phase=phase,
                identity=identity
                )



if __name__ == '__main__':
    main()



