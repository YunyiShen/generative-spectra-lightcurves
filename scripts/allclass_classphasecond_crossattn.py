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

train_data = np.load("../data/training_simulated_data.npz")
class_encoding = json.load(open('../data/train_simulated_class_dict.json'))

fluxes_std,  fluxes_mean = train_data['flux_std'], train_data['flux_mean']
wavelengths_std, wavelengths_mean = train_data['wavelength_std'], train_data['wavelength_mean']
spectime_std, spectime_mean = train_data['spectime_std'], train_data['spectime_mean'] 
concat = True

score_dict = {
            "d_model": 512,
            "d_mlp": 512,
            "n_layers": 4,
            "n_heads": 4,
            "concat_conditioning": concat,
            "crossattn": True,
        }
vdm = classtimecondVariationalDiffusionModel(d_feature=1, 
                                         noise_scale=1e-4, 
                                         noise_schedule="learned_linear",
                                         num_classes=len(class_encoding),
                                         score_dict = score_dict,
                                         )
flux, wavelength, mask = train_data['flux'], train_data['wavelength'], train_data['mask'] 
type, phase = train_data['type'], train_data['phase'] 
init_rngs = {'params': jax.random.key(0), 'sample': jax.random.key(1)}
out, params = vdm.init_with_output(init_rngs, flux[:2, :, None], wavelength[:2, :, None], phase[:2], type[:2], mask[:2])

with open(f'../ckpt/pretrain_classphasecond_static_dict_param_phase0_crossattn', 'rb') as f:
    serialized_model = f.read()

params = flax.serialization.from_bytes(params, serialized_model)

all_types = ['SN II',  'SN Ia', 'SN Ib', 'SN Ic']
wavelength_cond = (np.linspace(3000., 9000., flux.shape[1])[None, ...] - wavelengths_mean) / wavelengths_std 
#phase_cond = np.array([0.])
#phases = (0.0 - spectime_mean)/spectime_std
#breakpoint()
n_samples = 50
fig, ax1 = plt.subplots(1, 1, figsize=(7, 6))
'''
sample = classtimecondgenerate(vdm, params, jax.random.PRNGKey(42), 
                            (n_samples, len(wavelength_cond[0])), 
                            wavelength_cond[..., None], 
                            phase_cond,
                            type_cond,
                            np.ones_like(wavelength_cond), steps=200)
samples = sample.mean()[:,:,0]
np.save(f"../samples/II_phase{phase}.npy", samples)
for i in range(n_samples):
    plt.plot(wavelength_cond[0] * wavelengths_std + wavelengths_mean, 
             samples[i] * fluxes_std + fluxes_mean) # Generated sample

#plt.yscale("log")
plt.title("Generated spectra")
plt.savefig('../samples/II_samples_phase0.png')  
plt.close()
'''


camera = Camera(fig)
#breakpoint()
from tqdm import tqdm
sss = 0
#breakpoint()
for thistype in tqdm(all_types):
    phase_cond = np.array([(0.0 - spectime_mean)/spectime_std])
    type_cond = np.array([class_encoding[thistype]])
    sample = classtimecondgenerate(vdm, params, jax.random.PRNGKey(12345+sss), 
                            (n_samples, len(wavelength_cond[0])), 
                            wavelength_cond[..., None], 
                            phase_cond,
                            type_cond,
                            np.ones_like(wavelength_cond), steps=200)
    samples = sample.mean()[:,:,0]
    sss += 1
    np.save(f"../samples/simu_classphasecond_class{thistype}_crossattn.npy", samples)

    for i in range(10):
        plt.plot(wavelength_cond[0] * wavelengths_std + wavelengths_mean, 
             samples[i, :] * fluxes_std + fluxes_mean) # Generated sample
    plt.title(f"Generated {thistype} spectra")
    plt.savefig(f'{thistype}_samples_phase0_crossattn.png')  
    plt.close()






