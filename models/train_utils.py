import jax
import jax.numpy as np
import flax
from ml_collections import ConfigDict

from functools import partial
from tqdm import trange
replicate = flax.jax_utils.replicate
unreplicate = flax.jax_utils.unreplicate


def create_input_iter(ds):
    """Create an input iterator that prefetches to device."""

    def _prepare(xs):
        def _f(x):
            x = x._numpy()
            return x

        return jax.tree_util.tree_map(_f, xs)

    it = map(_prepare, ds)
    it = flax.jax_utils.prefetch_to_device(it, 2)
    return it


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def train_step(
    state, batch, rng, model, loss_fn, unconditional_dropout=False, p_uncond=0.0
):
    """Train for a single step."""
    x, conditioning, mask = batch

    # Unconditional dropout rng
    rng, rng_uncond = jax.random.split(rng)

    # Set a fraction p_uncond of conditioning vectors to zero if unconditional_dropout is True
    if conditioning is not None and unconditional_dropout:
        random_nums = jax.random.uniform(rng_uncond, conditioning.shape[:1]).reshape(
            -1, 1
        )
        conditioning = np.where(
            random_nums < p_uncond, np.zeros_like(conditioning), conditioning
        )

    loss, grads = jax.value_and_grad(loss_fn)(
        state.params, model, rng, x, conditioning, mask
    )
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    metrics = {"loss": jax.lax.pmean(loss, "batch")}
    return new_state, metrics


def param_count(pytree):
    """Count the number of parameters in a pytree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(pytree))


def to_wandb_config(d: ConfigDict, parent_key: str = "", sep: str = "."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, ConfigDict):
            items.extend(to_wandb_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def loss_vdm(outputs, masks=None):
    loss_diff, loss_klz, loss_recon = outputs
    if masks is None:
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 200, 1))
        masks = np.ones(x.shape[:-1])

    loss_batch = (((loss_diff + loss_klz) * masks[:, :, None]).sum((-1, -2)) + (loss_recon * masks[:, :, None]).sum((-1, -2))) / masks.sum(-1)
    
    return loss_batch.mean()

@partial(jax.pmap, axis_name="batch",)
def simplecond_train_step(state, flux, wavelength, phase,cond, masks, key_sample):
    
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

def simplecond_training_steps(pstate, flux, wavelength, phase, 
                             conditioning, mask, n_steps = 4000,
                             n_batch = 64,key = 0):
    # this can also be used for knob type of conditioning 
    key = jax.random.PRNGKey(key)
    num_local_devices = jax.local_device_count()
    print(f"{num_local_devices} GPUs available")
    with trange(n_steps) as steps:
        for step in steps:
            key, *train_step_key = jax.random.split(key, num=jax.local_device_count() + 1)  # Split key across devices
            train_step_key = np.asarray(train_step_key)

            idx = jax.random.choice(key, flux.shape[0], shape=(n_batch,))

            fluxes_batch, wavelength_batch, phase_batch,cond_batch, masks_batch = flux[idx], wavelength[idx], phase[idx], conditioning[idx], mask[idx]

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
        
            pstate, metrics = simplecond_train_step(pstate, fluxes_batch[..., None], wavelength_batch[..., None], phase_batch, cond_batch,masks_batch, train_step_key)
            #breakpoint()
            steps.set_postfix(loss=unreplicate(metrics["loss"]))
    
    return pstate, metrics




