import sys
sys.path.append("../")

import jax
import jax.numpy as np
import flax

from models.data_util import normalizing_spectra
from models.diffusion_cond import photometrycondVariationalDiffusionModel2
from models.diffusion_utils import photometrycondgenerate




train_data = np.load("../data/training_simulated_data_Ia.npz")

flux, wavelength, mask = train_data['flux'], train_data['wavelength'], train_data['mask'] 
type, phase = train_data['type'], train_data['phase'] 
photoflux, phototime, photomask = train_data['photoflux'], train_data['phototime'], train_data['photomask']
photowavelength = train_data['photowavelength']
photowavelength = (photowavelength - 0.62021224 >= 0.001) * 1 + (photowavelength + 2.69023892 >= 0.0001) * 1


fluxes_std,  fluxes_mean = train_data['flux_std'], train_data['flux_mean']
wavelengths_std, wavelengths_mean = train_data['wavelength_std'], train_data['wavelength_mean']


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
                                         )

init_rngs = {'params': jax.random.key(0), 'sample': jax.random.key(1)}
out, params = vdm.init_with_output(init_rngs, flux[:2, :, None], 
                                   wavelength[:2, :, None], 
                                   phase[:2], 
                                   mask[:2],
                                   photoflux[:2, :, None],
                                   phototime[:2, :, None],
                                   photowavelength[:2, :],
                                   photomask[:2],
                                   )


with open(f'../ckpt/pretrain_photometrycond_static_dict_param_cross_attn_Ia', 'rb') as f:
    serialized_model = f.read()
params = flax.serialization.from_bytes(params, serialized_model)

#test_data = np.load("../data/testing_simulated_data_Ia.npz")
test_data = np.load("../data/training_simulated_data_Ia.npz")
#test_data = np.load("../data/test_data_align_with_simu.npz")
#keep = np.array([x in Ias for x in test_data['type']])



photoflux, phototime, photomask = test_data['photoflux'][:,:], test_data['phototime'][:,:], test_data['photomask'][:,:]
photowavelength = test_data['photowavelength'][:,:]
photowavelength = (photowavelength - 0.62021224 >= 0.001) * 1 + (photowavelength + 2.69023892 >= 0.0001) * 1

first = 10
n_test_data = min(photoflux.shape[0], first)

n_samples = 50
wavelength_cond = (np.linspace(3000., 8000., flux.shape[1])[None, ...] - wavelengths_mean) / wavelengths_std
wavelength_cond = np.repeat(wavelength_cond, n_test_data, axis=0)

#posterior_samples = []
spectime_mean, spectime_std = train_data['spectime_mean'], train_data['spectime_std']
phase_cond = np.array([0.0 - spectime_mean] * n_samples * n_test_data) / spectime_std

#from tqdm import tqdm
#for i in tqdm(range(n_samples)):

    
photoflux, phototime, photomask = test_data['photoflux'][:n_test_data], test_data['phototime'][:n_test_data], test_data['photomask'][:n_test_data]
photowavelength = test_data['photowavelength'][:n_test_data]
photowavelength = (photowavelength - 0.62021224 >= 0.001) * 1 + (photowavelength + 2.69023892 >= 0.0001) * 1



photoflux = np.repeat(photoflux, n_samples, axis = 0)
phototime = np.repeat(phototime, n_samples, axis = 0)
photomask = np.repeat(photomask, n_samples, axis = 0)

photowavelength = np.repeat(photowavelength, n_samples, axis = 0)

wavelength_cond = (np.linspace(3000., 8000., flux.shape[1])[None, ...] - wavelengths_mean) / wavelengths_std
wavelength_cond = np.repeat(wavelength_cond, n_test_data * n_samples, axis=0)

spectime_mean, spectime_std = test_data['spectime_mean'], test_data['spectime_std']
phase_cond = np.array([0.0 - spectime_mean] * n_samples * n_test_data) / spectime_std
    


#breakpoint()
#breakpoint()
sample = photometrycondgenerate(vdm, params, jax.random.PRNGKey(42), 
                            (n_test_data * n_samples, len(wavelength_cond[0])), 
                            wavelength_cond[..., None], 
                            phase_cond,
                            np.ones_like(wavelength_cond),
                            photoflux[...,None],
                            phototime[...,None],
                            photowavelength,
                            photomask,
                            steps=200,
                            )
sample = sample.mean()[:,:,0]
sample = normalizing_spectra(sample)
#breakpoint()

posterior_samples = sample#.reshape(n_test_data, sample.shape[1],n_samples)
np.savez(f"../samples/posterior_test_photometrycond_first{n_test_data}_Ia.npz", 
         posterior_samples=posterior_samples, 
         gt = normalizing_spectra(test_data['flux'][:n_test_data]),
         SNtype = test_data['type'][:n_test_data],
         )

