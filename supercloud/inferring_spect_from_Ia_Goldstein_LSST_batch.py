import sys
sys.path.append("../")

import jax
import jax.numpy as np
import flax

from models.data_util import normalizing_spectra
from models.diffusion_cond import photometrycondVariationalDiffusionModel2
from models.diffusion_utils import photometrycondgenerate

midfilt = 3
centering = False
realistic = "realistic" 
#print(midfilt, centering, realistic)

def main():
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    all_starting_points = [i for i in range(num_tasks)]
    starting_points = all_starting_points[ my_task_id:len(all_starting_points):num_tasks]
    print(f"running inference using data starting from {starting_points}")






    all_data = np.load(f"../data/goldstein_processed/preprocessed_midfilt_{midfilt}_centering{centering}_{realistic}LSST_phase.npz")
    #breakpoint()
    training_idx = all_data['training_idx']
    testing_idx = all_data['testing_idx']



    flux, wavelength, mask = all_data['flux'][testing_idx,:], all_data['wavelength'][testing_idx,:], all_data['mask'][testing_idx,:]
    phase = all_data['phase'][testing_idx] 
    photoflux, phototime, photomask = all_data['photoflux'][testing_idx,:], all_data['phototime'][testing_idx,:], all_data['photomask'][testing_idx,:]
    #phototime = np.concatenate( (phototime, phototime, phototime), 1) # temp fix
    photowavelength = np.astype( all_data['photowavelength'][testing_idx,:], int)

    fluxes_std,  fluxes_mean = all_data['flux_std'], all_data['flux_mean']
    wavelengths_std, wavelengths_mean = all_data['wavelength_std'], all_data['wavelength_mean']
    phase_std, phase_mean = all_data['phase_std'], all_data['phase_mean']
    phototime_std, phototime_mean = all_data['phototime_std'], all_data['phototime_mean']
    photoflux_std, photoflux_mean = all_data['photoflux_std'], all_data['photoflux_mean']
    identity = all_data['identity'][testing_idx]

    wavelength_cond = np.copy(wavelength[:2])#(np.linspace(3000., 8000., flux.shape[1])[None, ...] - wavelengths_mean) / wavelengths_std
    phase_cond = np.copy(phase[:2])
    phototime_cond = np.copy(phototime[:2])

    # Define the model
    concat = True
    score_dict = {
            "d_model": 256,
            "d_mlp": 512,
            "n_layers": 6,
            "n_heads": 4,
            "concat_wavelength": concat,
        }
    vdm = photometrycondVariationalDiffusionModel2(d_feature=1, d_t_embedding=32, 
                                         noise_scale=1e-4, 
                                         noise_schedule="learned_linear",                                         
                                         score_dict = score_dict,
                                         nbands = 6,
                                         )

    init_rngs = {'params': jax.random.key(0), 'sample': jax.random.key(1)}
    out, params = vdm.init_with_output(init_rngs, flux[:2, :, None], 
                                   wavelength_cond[:2, :, None], 
                                   phase_cond[:2], 
                                   mask[:2],
                                   photoflux[:2, :, None],
                                   phototime_cond[:2, :, None],
                                   photowavelength[:2, :],#, None],
                                   photomask[:2],
                                   )

#breakpoint()
    with open(f'../ckpt/pretrain_photometrycond_static_dict_param_cross_attn_Ia_goldstein_midfilt_{midfilt}_centering{centering}_{realistic}LSST_phase', 'rb') as f:
        serialized_model = f.read()
    params = flax.serialization.from_bytes(params, serialized_model)


    n_batch = 15
    batch_size = 10
    batch_size = min(photoflux.shape[0], batch_size)

    total_SNs_per_job = n_batch * batch_size/5 # 5 phases

    n_samples = 50
    #posterior_samples = []

    for start in starting_points:

        offset = start * total_SNs_per_job
        from tqdm import tqdm
        for i in tqdm(range(n_batch)):
            phase_cond = np.repeat(phase[(i*batch_size + offset):((i+1) * batch_size + offset) ], n_samples, axis = 0) #np.array([0.0 ] * n_samples * batch_size)
            #breakpoint()
            #from tqdm import tqdm
            #for i in tqdm(range(n_samples)):

            photoflux_cond = np.repeat(photoflux[(i*batch_size + offset):((i+1) * batch_size + offset)], n_samples, axis = 0)
            phototime_cond = np.repeat(phototime[(i*batch_size + offset):((i+1) * batch_size + offset)], n_samples, axis = 0)
            photomask_cond = np.repeat(photomask[(i*batch_size + offset):((i+1) * batch_size + offset)], n_samples, axis = 0)

            photowavelength_cond = np.repeat( photowavelength[(i*batch_size + offset):((i+1) * batch_size + offset)], n_samples, axis = 0)

            wavelength_cond = wavelength[0,:][None, ...]#(np.linspace(3000., 8000., flux.shape[1])[None, ...] - wavelengths_mean) / wavelengths_std
            wavelength_cond = np.repeat(wavelength_cond, batch_size * n_samples, axis=0)

    
            gts = flux
            wavelengths = wavelength
            mask = mask

            #breakpoint()
            #print(phototime_cond[:10])
            sample = photometrycondgenerate(vdm, params, jax.random.PRNGKey(42), 
                            (batch_size * n_samples, len(wavelength_cond[0])), 
                            wavelength_cond[..., None], 
                            phase_cond,
                            np.ones_like(wavelength_cond),
                            photoflux_cond[...,None],
                            phototime_cond[...,None],
                            photowavelength_cond,#[...,None],
                            photomask_cond,
                            steps=200,
                            )
            sample = sample.mean()[:,:,0]
            #sample = normalizing_spectra(sample)
            #breakpoint()



            posterior_samples = sample#.reshape(batch_size, sample.shape[1],n_samples)
            np.savez(f"./samples/posterior_test_photometrycond_job{my_task_id}_batch{i}_size{batch_size}_Ia_Goldstein_centering{centering}_{realistic}LSST_phase.npz", 
                    posterior_samples=posterior_samples * fluxes_std + fluxes_mean, 
                    gt = gts[(i*batch_size + offset):((i+1) * batch_size + offset)]* fluxes_std + fluxes_mean,
                    wavelength = wavelengths[(i*batch_size + offset):((i+1) * batch_size + offset)] * wavelengths_std + wavelengths_mean,
                    mask = mask[(i*batch_size + offset):((i+1) * batch_size + offset)],
                    phase = phase[(i*batch_size + offset):((i+1) * batch_size + offset)] * phase_std + phase_mean,
                    photoflux = photoflux[(i*batch_size + offset):((i+1) * batch_size + offset)] * photoflux_std + photoflux_mean,
                    phototime = phototime[(i*batch_size + offset):((i+1) * batch_size + offset)] * phototime_std + phototime_mean,
                    photomask = photomask[(i*batch_size + offset):((i+1) * batch_size + offset)],
                    photoband = photowavelength[(i*batch_size + offset):((i+1) * batch_size + offset)],
                    identity = identity[(i*batch_size + offset):((i+1) * batch_size + offset)],
                    )

if __name__ == "__main__":
    main()
