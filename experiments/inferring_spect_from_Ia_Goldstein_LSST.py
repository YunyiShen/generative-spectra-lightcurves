import sys
sys.path.append("../")

import jax
import jax.numpy as np
import flax

from models.data_util import normalizing_spectra
from models.diffusion_cond import photometrycondVariationalDiffusionModel2
from models.diffusion_utils import photometrycondgenerate
midfilt = 3
centering = True

all_data = np.load(f"../data/goldstein_processed/preprocessed_midfilt_{midfilt}_centering{centering}_LSST_phase.npz")
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
wavelength_cond = np.copy(wavelength[:2])#(np.linspace(3000., 8000., flux.shape[1])[None, ...] - wavelengths_mean) / wavelengths_std
phase_cond = np.copy(phase[:2])

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
                                   phototime[:2, :, None],
                                   photowavelength[:2, :],#, None],
                                   photomask[:2],
                                   )

#breakpoint()
with open(f'../ckpt/pretrain_photometrycond_static_dict_param_cross_attn_Ia_goldstein_midfilt_{midfilt}_centering{centering}_LSST_phase', 'rb') as f:
    serialized_model = f.read()
params = flax.serialization.from_bytes(params, serialized_model)


first = 10
n_test_data = min(photoflux.shape[0], first)

n_samples = 50
#posterior_samples = []
phase_cond = np.repeat(phase[:n_test_data], n_samples, axis = 0) #np.array([0.0 ] * n_samples * n_test_data)
#breakpoint()
#from tqdm import tqdm
#for i in tqdm(range(n_samples)):

photoflux = np.repeat(photoflux[:n_test_data], n_samples, axis = 0)
phototime = np.repeat(phototime[:n_test_data], n_samples, axis = 0)
photomask = np.repeat(photomask[:n_test_data], n_samples, axis = 0)

photowavelength = np.repeat( photowavelength[:n_test_data], n_samples, axis = 0)

wavelength_cond = wavelength[0,:][None, ...]#(np.linspace(3000., 8000., flux.shape[1])[None, ...] - wavelengths_mean) / wavelengths_std
wavelength_cond = np.repeat(wavelength_cond, n_test_data * n_samples, axis=0)

    
gts = flux
wavelengths = wavelength
mask = mask

#breakpoint()
sample = photometrycondgenerate(vdm, params, jax.random.PRNGKey(42), 
                            (n_test_data * n_samples, len(wavelength_cond[0])), 
                            wavelength_cond[..., None], 
                            phase_cond,
                            np.ones_like(wavelength_cond),
                            photoflux[...,None],
                            phototime[...,None],
                            photowavelength,#[...,None],
                            photomask,
                            steps=200,
                            )
sample = sample.mean()[:,:,0]
#sample = normalizing_spectra(sample)
#breakpoint()



posterior_samples = sample#.reshape(n_test_data, sample.shape[1],n_samples)
np.savez(f"../samples/posterior_test_photometrycond_first{n_test_data}_Ia_Goldstein_centering{centering}_LSST_phase.npz", 
         posterior_samples=posterior_samples, 
         gt = gts[:n_test_data],
         wavelength = wavelengths[:n_test_data] * wavelengths_std + wavelengths_mean,
         mask = mask[:n_test_data],
         )

