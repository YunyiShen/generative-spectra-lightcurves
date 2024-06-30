import sys
sys.path.append("../")

from models.gp import gp_peak_time
import jax.numpy as np
import numpy as onp

random = onp.random.default_rng(135)
time = onp.sort(random.uniform(0, 10, 50))

flux = np.exp(-0.5 * np.square((time - 5) / np.exp(0.5))) + random.normal(0, 0.1, 50)

peak = gp_peak_time(time, flux)
print(peak)