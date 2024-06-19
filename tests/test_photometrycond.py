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

from models.data_util import specdata
from models.diffusion_cond import photometrycondVariationalDiffusionModel

replicate = flax.jax_utils.replicate
unreplicate = flax.jax_utils.unreplicate


spec_data = specdata(master_list = "../data/ZTFBTS/ZTFBTS_TransientTable_test.csv",
                     verbose = False)
flux, freq, mask, type,redflux, redtime, redmask, greentime, greenflux, greenmask = spec_data.get_data()

# Define the model
vdm = photometrycondVariationalDiffusionModel(d_feature=1, d_t_embedding=32, 
                                         noise_scale=1e-4, 
                                         noise_schedule="learned_linear",)

init_rngs = {'params': jax.random.key(0), 'sample': jax.random.key(1)}
out, params = vdm.init_with_output(init_rngs, flux[:2, :, None], freq[:2, :, None], mask[:2], 
                                   greenflux[:2, :, None], greentime[:2, :, None], greenmask[:2], 
                                   redtime[:2, :, None], redflux[:2, :, None], redmask[:2])

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
def train_step(state, flux, freq, masks, 
               green_flux, green_time, green_mask, 
               red_time, red_flux, red_mask, key_sample):
    
    def loss_fn(params):
        outputs = state.apply_fn(params, flux, freq, masks, 
                                 green_flux, green_time, green_mask, 
                                 red_time, red_flux, red_mask,
                                 rngs={"sample": key_sample})
        loss = loss_vdm(outputs, masks)
        
        return loss

    # Get loss, grads, and update state
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    metrics = {"loss": jax.lax.pmean(loss, "batch")}
    return new_state, metrics

n_steps = 300
n_batch = 32

key = jax.random.PRNGKey(0)
num_local_devices = jax.local_device_count()
print(f"{num_local_devices} GPUs available")
with trange(n_steps) as steps:
    for step in steps:
        key, *train_step_key = jax.random.split(key, num=jax.local_device_count() + 1)  # Split key across devices
        train_step_key = np.asarray(train_step_key)

        idx = jax.random.choice(key, flux.shape[0], shape=(n_batch,))

        fluxes_batch, freq_batch, masks_batch = flux[idx], freq[idx], mask[idx]
        green_fluxes_batch, green_time_batch, green_masks_batch = greenflux[idx], greentime[idx], greenmask[idx]
        red_fluxes_batch, red_time_batch, red_masks_batch = redflux[idx], redtime[idx], redmask[idx]

        # Split batches across devices
        fluxes_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), fluxes_batch)
        freq_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), freq_batch)
        masks_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), masks_batch)

        green_fluxes_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), green_fluxes_batch)
        green_time_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), green_time_batch)
        green_masks_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), green_masks_batch)

        red_fluxes_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), red_fluxes_batch)
        red_time_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), red_time_batch)
        red_masks_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), red_masks_batch)


        # Convert to np.ndarray
        fluxes_batch = np.array(fluxes_batch)
        freq_batch = np.array(freq_batch)
        masks_batch = np.array(masks_batch)

        green_fluxes_batch = np.array(green_fluxes_batch)
        green_time_batch = np.array(green_time_batch)
        green_masks_batch = np.array(green_masks_batch)

        red_fluxes_batch = np.array(red_fluxes_batch)
        red_time_batch = np.array(red_time_batch)
        red_masks_batch = np.array(red_masks_batch)
        
        pstate, metrics = train_step(pstate, fluxes_batch[..., None], 
                                     freq_batch[..., None], masks_batch, 
                                     green_fluxes_batch[..., None],
                                     green_time_batch[..., None],
                                     green_masks_batch,
                                     red_time_batch[..., None],
                                     red_fluxes_batch[..., None],
                                     red_masks_batch,
                                     train_step_key)
        #breakpoint()
        steps.set_postfix(loss=unreplicate(metrics["loss"]))

### test for generating
from models.diffusion_utils import photometrycondgenerate

# Generate samples
n_samples = 24
freq_cond = freq[:1, : np.sum(mask[0])]
freq_cond = np.linspace(np.min(freq_cond), np.max(freq_cond), 214)[None, ...]

green_flux_cond = greenflux[:1, :]
green_time_cond = greentime[:1, :]
green_mask_cond = greenmask[:1, :]
red_flux_cond = redflux[:1, :]
red_time_cond = redtime[:1, :]
red_mask_cond = redmask[:1, :]




samples = photometrycondgenerate(vdm, unreplicate(pstate).params, 
                            jax.random.PRNGKey(412141), 
                            (n_samples, len(freq_cond[0])), 
                            freq_cond[..., None], 
                            np.ones_like(freq_cond), 
                            green_flux_cond[..., None],
                            green_time_cond[..., None],
                            green_mask_cond,
                            red_time_cond[..., None],
                            red_flux_cond[..., None],
                            red_mask_cond,
                            steps=200)

np.save("photometry_samples.npy", samples)

import matplotlib.pyplot as plt
for i in range(n_samples):
    plt.plot(freq_cond[0] * spec_data.wavelengths_std + spec_data.wavelengths_mean, 
             samples.mean()[i, :, 0] * spec_data.fluxes_std + spec_data.fluxes_mean) # Generated sample

plt.yscale("log")
plt.title("Generated spectra")
plt.savefig('photometry__samples.png')  # You can specify different file formats like 'plot.pdf', 'plot.jpg', etc.

# Optionally, close the plot to free memory
plt.close()

