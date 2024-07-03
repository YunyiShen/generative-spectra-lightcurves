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
import json

from models.data_util import specdata


spec_data = specdata(master_list = "../data/ZTFBTS/ZTFBTS_TransientTable_test_dataloader.csv",
                     verbose = True,
                     spectime = True)
flux, freq, mask, type, phase, photoflux, phototime, photomask = spec_data.get_data(concat_photometry = True)
np.savez("../data/test_data.npz", 
         flux=flux, 
         freq=freq, 
         mask=mask, 
         type=type, 
         phase=phase, photoflux=photoflux, 
         phototime=phototime, photomask=photomask)
with open('../class_dict.json', 'w') as fp:
    json.dump(spec_data.class_encoding, fp)


breakpoint()