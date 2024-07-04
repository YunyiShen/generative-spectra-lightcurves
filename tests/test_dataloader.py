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
flux, wavelength, mask, type, phase, photoflux, phototime, photowavelength, photomask = spec_data.get_data(concat_photometry = True)

breakpoint()
np.savez("../data/test_dataloader.npz", 
         flux=flux, 
         wavelength=wavelength, 
         mask=mask, 
         type=type, 
         phase=phase, photoflux=photoflux, 
         phototime=phototime, photomask=photomask)
with open('../test_dataloader_class_dict.json', 'w') as fp:
    json.dump(spec_data.class_encoding, fp)


breakpoint()