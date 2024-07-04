import sys
sys.path.append("../")

import jax
import jax.numpy as np
import flax
import flax.linen as nn
from tqdm import trange
from celluloid import Camera
from matplotlib import pyplot as plt

from models.diffusion_cond import classtimecondVariationalDiffusionModel
from models.diffusion_utils import classtimecondgenerate


import json

train_data = np.load("../data/train_data.npz")
class_encoding = json.load(open('../data/train_class_dict.json'))

fluxes_std,  fluxes_mean = train_data['flux_std'], train_data['flux_mean']
wavelengths_std, wavelengths_mean = train_data['wavelength_std'], train_data['wavelength_mean']
spectime_std, spectime_mean = train_data['spectime_std'], train_data['spectime_mean'] 

score_dict = {
            "d_model": 256,
            "d_mlp": 512,
            "n_layers": 4,
            "n_heads": 4,
            "concat_conditioning": False,
        }
vdm = classtimecondVariationalDiffusionModel(d_feature=1, d_t_embedding=32, 
                                         noise_scale=1e-4, 
                                         noise_schedule="learned_linear",
                                         num_classes=len(class_encoding),
                                         score_dict = score_dict,
                                         )
flux, wavelength, mask = train_data['flux'], train_data['wavelength'], train_data['mask'] 
type, phase = train_data['type'], train_data['phase'] 
init_rngs = {'params': jax.random.key(0), 'sample': jax.random.key(1)}
out, params = vdm.init_with_output(init_rngs, flux[:2, :, None], wavelength[:2, :, None], phase[:2], type[:2], mask[:2])

with open('../ckpt/calssphasecond_static_dict_param_phase0', 'rb') as f:
    serialized_model = f.read()

params = flax.serialization.from_bytes(params, serialized_model)


wavelength_cond = (np.linspace(3000., 8000., flux.shape[1])[None, ...] - wavelengths_mean) / wavelengths_std 
type_cond = np.array([class_encoding['SN Ia']])
phases = (np.linspace(0, 30, 31) - spectime_mean)/spectime_std
#breakpoint()
n_samples = 50
fig, ax1 = plt.subplots(1, 1, figsize=(7, 6))
camera = Camera(fig)
#breakpoint()
from tqdm import tqdm
for phase in tqdm(phases):
    phase_cond = np.array([phase])
    sample = classtimecondgenerate(vdm, params, jax.random.PRNGKey(42), 
                            (n_samples, len(wavelength_cond[0])), 
                            wavelength_cond[..., None], 
                            phase_cond,
                            type_cond,
                            np.ones_like(wavelength_cond), steps=200)
    samples = sample.mean()[:,:,0]
    np.save(f"../samples/classphasecond_phase{phase}.npy", samples)
    samples_plot = np.median(samples, axis=0)
    ax1.set_ylim(0, 1.)
    ax1.plot(wavelength_cond[0] * wavelengths_std + wavelengths_mean, 
             samples_plot * fluxes_std + fluxes_mean, color='black', alpha=1)
    camera.snap()
animation = camera.animate()
animation.save('../samples/classphasecond.mp4')






