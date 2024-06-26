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


from models.data_util import specdata
from models.diffusion_cond import classcondVariationalDiffusionModel
from models.transformer import Transformer

replicate = flax.jax_utils.replicate
unreplicate = flax.jax_utils.unreplicate


spec_data = specdata(master_list = "../data/ZTFBTS/ZTFBTS_TransientTable_train.csv",
                     verbose = False)
flux, wavelength, mask, type,redflux, redtime, redmask, greentime, greenflux, greenmask = spec_data.get_data()

# Define the model
score_dict = {
            "d_model": 256,
            "d_mlp": 512,
            "n_layers": 4,
            "n_heads": 4,
            "concat_conditioning": False,
        }
vdm = classcondVariationalDiffusionModel(d_feature=1, d_t_embedding=32, 
                                         noise_scale=1e-4, 
                                         noise_schedule="learned_linear",
                                         num_classes=spec_data.num_class,
                                         score_dict = score_dict,
                                         )

init_rngs = {'params': jax.random.key(0), 'sample': jax.random.key(1)}
out, params = vdm.init_with_output(init_rngs, flux[:2, :, None], wavelength[:2, :, None], type[:2], mask[:2])

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=3e-4,
    warmup_steps=500,
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
def train_step(state, flux, wavelength, cond, masks, key_sample):
    
    def loss_fn(params):
        outputs = state.apply_fn(params, flux, wavelength, cond,masks, rngs={"sample": key_sample})
        loss = loss_vdm(outputs, masks)
        
        return loss

    # Get loss, grads, and update state
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    metrics = {"loss": jax.lax.pmean(loss, "batch")}
    return new_state, metrics

n_steps = 6000
n_batch = 32

key = jax.random.PRNGKey(0)
num_local_devices = jax.local_device_count()
print(f"{num_local_devices} GPUs available")
with trange(n_steps) as steps:
    for step in steps:
        key, *train_step_key = jax.random.split(key, num=jax.local_device_count() + 1)  # Split key across devices
        train_step_key = np.asarray(train_step_key)

        idx = jax.random.choice(key, flux.shape[0], shape=(n_batch,))

        fluxes_batch, wavelength_batch, cond_batch, masks_batch = flux[idx], wavelength[idx], type[idx], mask[idx]

        # Split batches across devices
        fluxes_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), fluxes_batch)
        wavelength_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), wavelength_batch)
        cond_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), cond_batch)
        masks_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), masks_batch)

        # Convert to np.ndarray
        fluxes_batch = np.array(fluxes_batch)
        wavelength_batch = np.array(wavelength_batch)
        cond_batch = np.array(cond_batch)
        masks_batch = np.array(masks_batch)
        
        pstate, metrics = train_step(pstate, fluxes_batch[..., None], wavelength_batch[..., None], cond_batch,masks_batch, train_step_key)
        #breakpoint()
        steps.set_postfix(loss=unreplicate(metrics["loss"]))

### test for generating
from models.diffusion_utils import classcondgenerate

# Generate samples
n_samples = 100
wavelength_cond = wavelength[:1, : np.sum(mask[0])]
wavelength_cond = np.linspace(np.min(wavelength_cond), np.max(wavelength_cond), 214)[None, ...]
type_cond = np.array([spec_data.class_encoding['SN Ia']])



samples = classcondgenerate(vdm, unreplicate(pstate).params, 
                            jax.random.PRNGKey(412141), 
                            (n_samples, len(wavelength_cond[0])), 
                            wavelength_cond[..., None], 
                            type_cond,
                            np.ones_like(wavelength_cond), steps=200)

np.save("Ia_samples_sine.npy", samples.mean()[:, :, 0] * spec_data.fluxes_std + spec_data.fluxes_mean)
np.save("Ia_wavelength_sine.npy", wavelength_cond[0]* spec_data.wavelengths_std + spec_data.wavelengths_mean)

import matplotlib.pyplot as plt
for i in range(n_samples):
    plt.plot(wavelength_cond[0] * spec_data.wavelengths_std + spec_data.wavelengths_mean, 
             samples.mean()[i, :, 0] * spec_data.fluxes_std + spec_data.fluxes_mean) # Generated sample

#plt.yscale("log")
plt.title("Generated spectra")
plt.savefig('Ia_samples_sine.png')  
plt.close()

# save parameters
np.save("../ckpt/calsscond_static_dict_param_sine",unreplicate(pstate).params)
# this is not an elegant solution but I cannot save checkpoint on supercloud



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