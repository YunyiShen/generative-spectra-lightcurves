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


spec_data = specdata(master_list = "../data/ZTFBTS/ZTFBTS_TransientTable_test.csv",
                     verbose = True)
flux, freq, mask, type,redflux, redtime, redmask, greentime, greenflux, greenmask = spec_data.get_data()
breakpoint()