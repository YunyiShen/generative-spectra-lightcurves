import pandas as pd
import numpy as np

# Load the master CSV file
data = pd.read_csv('../data/ZTFBTS/ZTFBTS_TransientTable_withpeak.csv')

# Shuffle the rows randomly
data_shuffled = data.sample(frac=1, random_state=42)  # Using a fixed random_state for reproducibility

# Calculate the number of rows for training and testing
total_rows = len(data_shuffled)
train_rows = int(total_rows * 3 / 4)
test_rows = total_rows - train_rows

# Split the shuffled data into training and testing sets
train_data = data_shuffled[:train_rows]
test_data = data_shuffled[train_rows:]

# Save the training and testing data to separate CSV files
train_data.to_csv('../data/ZTFBTS/ZTFBTS_TransientTable_withpeak_train.csv', index=False)
test_data.to_csv('../data/ZTFBTS/ZTFBTS_TransientTable_withpeak_test.csv', index=False)



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


spec_data = specdata(master_list = "../data/ZTFBTS/ZTFBTS_TransientTable_withpeak.csv",
                     verbose = True,
                     spectime = True)
flux, freq, mask, type, phase, photoflux, phototime, photomask = spec_data.get_data(concat_photometry = True)
np.savez("../data/all_data.npz", 
         flux=flux, 
         wavelength=freq, 
         mask=mask, 
         type=type, 
         phase=phase, photoflux=photoflux, 
         phototime=phototime, photomask=photomask,
         flux_mean=spec_data.fluxes_mean,
         flux_std=spec_data.fluxes_std,
         freq_mean=spec_data.wavelengths_mean,
         freq_std=spec_data.wavelengths_std,

        green_mean = spec_data.green_mean,
        green_std = spec_data.green_std,
        red_mean = spec_data.red_mean,
        red_std = spec_data.red_std,
        green_time_mean = spec_data.green_time_mean,
        green_time_std = spec_data.green_time_std,
        red_time_mean = spec_data.red_time_mean,
        red_time_std = spec_data.red_time_std,
        spectime_mean = spec_data.spectime_mean,
        spectime_std = spec_data.spectime_std)
with open('../data/all_class_dict.json', 'w') as fp:
    json.dump(spec_data.class_encoding, fp)

'''
class_encoding = spec_data.class_encoding

spec_data = specdata(master_list = "../data/ZTFBTS/ZTFBTS_TransientTable_withpeak_test.csv",
                     verbose = True,
                     spectime = True, 
                     class_encoding = class_encoding, 
                     z_score=False)
flux, freq, mask, type, phase, photoflux, phototime, photomask = spec_data.get_data(concat_photometry = True)
np.savez("../data/test_data.npz", 
         flux=flux, 
         freq=freq, 
         mask=mask, 
         type=type, 
         phase=phase, photoflux=photoflux, 
         phototime=phototime, photomask=photomask)

'''