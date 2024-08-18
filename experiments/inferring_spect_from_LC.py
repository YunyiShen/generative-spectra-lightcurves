import sys
sys.path.append("../")

import jax
import jax.numpy as np
import flax

from models.data_util import normalizing_spectra
from models.diffusion_cond import photometrycondVariationalDiffusionModel2
from models.diffusion_utils import photometrycondgenerate




train_data = np.load("../data/training_simulated_data.npz")

flux, wavelength, mask = train_data['flux'], train_data['wavelength'], train_data['mask'] 
type, phase = train_data['type'], train_data['phase'] 
photoflux, phototime, photomask = train_data['photoflux'], train_data['phototime'], train_data['photomask']
photowavelength = train_data['photowavelength']


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
                                   photowavelength[:2, :, None],
                                   photomask[:2],
                                   )


with open(f'../ckpt/pretrain_photometrycond_static_dict_param_cross_attn', 'rb') as f:
    serialized_model = f.read()
params = flax.serialization.from_bytes(params, serialized_model)

test_data = np.load("../data/testing_simulated_data.npz")
photoflux, phototime, photomask = test_data['photoflux'], test_data['phototime'], test_data['photomask']
photowavelength = test_data['photowavelength']

n_test_data = min(photoflux.shape[0], 5000)

n_samples = 100
wavelength_cond = (np.linspace(3000., 8000., flux.shape[1])[None, ...] - wavelengths_mean) / wavelengths_std
wavelength_cond = np.repeat(wavelength_cond, n_test_data, axis=0)

posterior_samples = []
spectime_mean, spectime_std = train_data['spectime_mean'], train_data['spectime_std']
phase_cond = np.array([0.0 - spectime_mean] * n_test_data) / spectime_std

from tqdm import tqdm
for i in tqdm(range(n_samples)):
    #breakpoint()
    sample = photometrycondgenerate(vdm, params, jax.random.PRNGKey(42 + i), 
                            (n_test_data, len(wavelength_cond[0])), 
                            wavelength_cond[:n_test_data][..., None], 
                            phase_cond[:n_test_data],
                            np.ones_like(wavelength_cond[:n_test_data]),
                            photoflux[:n_test_data][...,None],
                            phototime[:n_test_data][...,None],
                            photowavelength[:n_test_data][...,None],
                            photomask[:n_test_data],
                            steps=200,
                            )
    sample = sample.mean()[:,:,0]
    sample = normalizing_spectra(sample)
    posterior_samples.append(sample)

posterior_samples = np.stack(posterior_samples, axis=-1)
np.savez("../samples/posterior_test_photometrycond.npz", 
         posterior_samples=posterior_samples)

