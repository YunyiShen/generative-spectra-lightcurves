import sys
sys.path.append("../")

import jax
import jax.numpy as np
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
from functools import partial
import optax
from tqdm import trange
import pandas as pd

from flax.training import checkpoints, train_state
from flax import struct, serialization
import orbax.checkpoint


from models.diffusion_cond import classtimecondVariationalDiffusionModel2
import json

replicate = flax.jax_utils.replicate
unreplicate = flax.jax_utils.unreplicate

train_data = np.load("../data/training_simulated_data.npz")


flux, wavelength, mask = train_data['flux'], train_data['wavelength'], train_data['mask'] 
type, phase = train_data['type'], train_data['phase'] 
photoflux, phototime, photomask = train_data['photoflux'], train_data['phototime'], train_data['photomask']
class_encoding = json.load(open('../data/train_simulated_class_dict.json'))
spectime_mean, spectime_std = train_data['spectime_mean'], train_data['spectime_std']

fluxes_std,  fluxes_mean = train_data['flux_std'], train_data['flux_mean']
wavelengths_std, wavelengths_mean = train_data['wavelength_std'], train_data['wavelength_mean']
#breakpoint()

# Define the model
'''
# a seemingly not bad one
score_dict = {
            "d_model": 256,
            "d_mlp": 512,
            "n_layers": 8,
            "n_heads": 4,
        }
'''

score_dict = {
            "d_model": 512,
            "d_mlp": 512,
            "n_layers": 6,
            "n_heads": 4,
        }
vdm = classtimecondVariationalDiffusionModel2(d_feature=1,
                                              
                                         noise_scale=1e-4, 
                                         noise_schedule="learned_linear",
                                         num_classes=len(class_encoding),
                                         score_dict = score_dict,
                                         )

init_rngs = {'params': jax.random.key(0), 'sample': jax.random.key(1)}
out, params = vdm.init_with_output(init_rngs, flux[:2, :, None], wavelength[:2, :, None], phase[:2], type[:2], mask[:2])

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=3e-4,
    warmup_steps= 500,
    decay_steps=3000,
)

tx = optax.adamw(learning_rate=schedule, weight_decay=1e-5)
state = TrainState.create(apply_fn=vdm.apply, params=params, tx=tx)
pstate = replicate(state)

x = jax.random.normal(jax.random.PRNGKey(0), (1, 200, 1))
def loss_vdm(outputs, masks=None):
    loss_diff, loss_klz, loss_recon = outputs
    if masks is None:
        masks = np.ones(x.shape[:-1])

    loss_batch = (((loss_diff + loss_klz) * masks[:, :, None]).sum((-1, -2)) + (loss_recon * masks[:, :, None]).sum((-1, -2))) / masks.sum(-1)
    
    return loss_batch.mean()

@partial(jax.pmap, axis_name="batch",)
def train_step(state, flux, wavelength, phase,cond, masks, key_sample):
    
    def loss_fn(params):
        outputs = state.apply_fn(params, flux, wavelength, phase,cond,masks, rngs={"sample": key_sample})
        loss = loss_vdm(outputs, masks)
        
        return loss

    # Get loss, grads, and update state
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    metrics = {"loss": jax.lax.pmean(loss, "batch")}
    return new_state, metrics

n_steps = 4000
n_batch = 64

key = jax.random.PRNGKey(0)
num_local_devices = jax.local_device_count()
print(f"{num_local_devices} GPUs available")
with trange(n_steps) as steps:
    for step in steps:
        key, *train_step_key = jax.random.split(key, num=jax.local_device_count() + 1)  # Split key across devices
        train_step_key = np.asarray(train_step_key)

        idx = jax.random.choice(key, flux.shape[0], shape=(n_batch,))

        fluxes_batch, wavelength_batch, phase_batch,cond_batch, masks_batch = flux[idx], wavelength[idx], phase[idx], type[idx], mask[idx]

        # Split batches across devices
        fluxes_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), fluxes_batch)
        wavelength_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), wavelength_batch)
        phase_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), phase_batch)
        cond_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), cond_batch)
        masks_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), masks_batch)

        # Convert to np.ndarray
        fluxes_batch = np.array(fluxes_batch)
        wavelength_batch = np.array(wavelength_batch)
        phase_batch = np.array(phase_batch)
        cond_batch = np.array(cond_batch)
        masks_batch = np.array(masks_batch)
        
        pstate, metrics = train_step(pstate, fluxes_batch[..., None], wavelength_batch[..., None], phase_batch, cond_batch,masks_batch, train_step_key)
        #breakpoint()
        steps.set_postfix(loss=unreplicate(metrics["loss"]))

### test for generating
from models.diffusion_utils import classtimecondgenerate
# save parameters
# np.save("../ckpt/calssphasecond_static_dict_param_phase0",unreplicate(pstate).params)
# this is not an elegant solution but I cannot save checkpoint on supercloud
byte_output = serialization.to_bytes(unreplicate(pstate).params)
with open(f'../ckpt/pretrain_classphasecond_static_dict_param_phase0_crossattn2', 'wb') as f:
    f.write(byte_output)



from matplotlib import pyplot as plt
n_samples = 50
all_types = ['SN II',  'SN Ia', 'SN Ib', 'SN Ic']
wavelength_cond = (np.linspace(3000., 9000., flux.shape[1])[None, ...] - wavelengths_mean) / wavelengths_std 
wavelength_cond = np.repeat(wavelength_cond, n_samples, axis=0)#phase_cond = np.array([0.])
#phases = (0.0 - spectime_mean)/spectime_std
#breakpoint()
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
print("generate all classes...")

#breakpoint()
from tqdm import tqdm
sss = 0
#breakpoint()
for thistype in tqdm(all_types):
    phase_cond = np.array([(0.0 - spectime_mean)/spectime_std] * n_samples)
    type_cond = np.array([class_encoding[thistype]]* n_samples)
    sample = classtimecondgenerate(vdm, unreplicate(pstate).params, jax.random.PRNGKey(12345+sss), 
                            (n_samples, len(wavelength_cond[0])), 
                            wavelength_cond[..., None], 
                            phase_cond,
                            type_cond,
                            np.ones_like(wavelength_cond), steps=200)
    samples = sample.mean()[:,:,0]
    sss += 1
    np.save(f"../samples/simu_classphasecond_class{thistype}_crossattn2.npy", samples)

    for i in range(10):
        plt.plot(wavelength_cond[0] * wavelengths_std + wavelengths_mean, 
             samples[i, :] * fluxes_std + fluxes_mean) # Generated sample
    plt.title(f"Generated {thistype} spectra")
    plt.savefig(f'{thistype}_samples_phase0_crossattn2.png')  
    plt.close()






print("Done!")

'''
from flax.training import orbax_utils

ckpt = {'model': unreplicate(pstate)}

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
ckpt_dir = '../ckpt/class_cond_dashed'
import os
import shutil
if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)  # Remove any existing checkpoints from the last notebook run.
breakpoint()
orbax_checkpointer.save(os.path.abspath(ckpt_dir), ckpt, save_args=save_args)
'''