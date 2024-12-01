import sys
sys.path.append("../")

import jax
import jax.numpy as np
import flax

from models.data_util import normalizing_spectra
from models.diffusion_cond import photometrycondVariationalDiffusionModel2
from models.diffusion_utils import photometrycondgenerate
import json



train_data = np.load("../data/train_data_align_with_simu_minimal_centered.npz")
type = train_data['type']
## Ia subtypes in ZTF data: [4,7,8,13,15,17]
class_encoding = json.load(open('../data/train_class_dict_align_with_simu_centered.json'))
Ias = [ v for k, v in class_encoding.items() if "Ia" in k ]
keep = np.array([x in Ias for x in train_data['type']])

#breakpoint()
flux, wavelength, mask = train_data['flux'][keep,:], train_data['wavelength'][keep,:], train_data['mask'][keep,:]
type, phase = train_data['type'][keep], train_data['phase'][keep] 
photoflux, phototime, photomask = train_data['photoflux'][keep,:], train_data['phototime'][keep,:], train_data['photomask'][keep,:]
photowavelength = np.astype(  train_data['photowavelength'][keep,:], int)

ztfid = train_data['ztfid'][keep]

fluxes_std,  fluxes_mean = train_data['flux_std'], train_data['flux_mean']
wavelengths_std, wavelengths_mean = train_data['wavelength_std'], train_data['wavelength_mean']
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
                                         )

init_rngs = {'params': jax.random.key(0), 'sample': jax.random.key(1)}
out, params = vdm.init_with_output(init_rngs, flux[:2, :, None], 
                                   wavelength[:2, :, None], 
                                   phase_cond[:2], 
                                   mask[:2],
                                   photoflux[:2, :, None],
                                   phototime[:2, :, None],
                                   photowavelength[:2, :],#, None],
                                   photomask[:2],
                                   )


with open(f'../ckpt/pretrain_photometrycond_static_dict_param_cross_attn_Ia_ZTF_fintuning_centered', 'rb') as f:
    serialized_model = f.read()
params = flax.serialization.from_bytes(params, serialized_model)

test_data = np.load("../data/train_data_align_with_simu_minimal_centered.npz")
keep = np.array([x in Ias for x in test_data['type']])



photoflux, phototime, photomask = test_data['photoflux'][keep,:], test_data['phototime'][keep,:], test_data['photomask'][keep,:]
photowavelength = test_data['photowavelength'][keep,:]

first = 30
n_test_data = min(photoflux.shape[0], first)

n_samples = 50
wavelength_cond = (np.linspace(3000., 8000., flux.shape[1])[None, ...] - wavelengths_mean) / wavelengths_std
wavelength_cond = np.repeat(wavelength_cond, n_test_data, axis=0)

#posterior_samples = []
spectime_mean, spectime_std = train_data['spectime_mean'], train_data['spectime_std']
phase_cond = np.repeat(phase[:n_test_data], n_samples, axis = 0) #np.array([0.0 ] * n_samples * n_test_data)

#from tqdm import tqdm
#for i in tqdm(range(n_samples)):

    
photoflux, phototime, photomask = test_data['photoflux'][:n_test_data], test_data['phototime'][:n_test_data], test_data['photomask'][:n_test_data]
photowavelength = np.astype( test_data['photowavelength'][:n_test_data], int)
#breakpoint()
ztfid = test_data['ztfid'][:n_test_data]

photoflux = np.repeat(photoflux, n_samples, axis = 0)
phototime = np.repeat(phototime, n_samples, axis = 0)
photomask = np.repeat(photomask, n_samples, axis = 0)

photowavelength = np.repeat( photowavelength, n_samples, axis = 0)

wavelength_cond = (np.linspace(3000., 8000., flux.shape[1])[None, ...] - wavelengths_mean) / wavelengths_std
wavelength_cond = np.repeat(wavelength_cond, n_test_data * n_samples, axis=0)

spectime_mean, spectime_std = train_data['spectime_mean'], train_data['spectime_std']
phase_cond = np.repeat(phase[:n_test_data], n_samples, axis = 0) #np.array([0.0 ] * n_samples * n_test_data)
    
gts = test_data['flux'][keep,:]
wavelengths = test_data['wavelength'][keep,:]
mask = test_data['mask'][keep,:]
SNtype = test_data['type'][keep]    
ztfid = test_data['ztfid'][keep]

#breakpoint()
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
np.savez(f"../samples/posterior_test_photometrycond_first{n_test_data}_memory_Ia_ZTF_centered.npz", 
         posterior_samples=posterior_samples, 
         gt = gts[:n_test_data],
         wavelength = wavelengths[:n_test_data] * wavelengths_std + wavelengths_mean,
         mask = mask[:n_test_data],
         SNtype = SNtype[:n_test_data],
         ztfid = ztfid[:n_test_data],
         )

