import pandas as pd
import numpy as np

# fix random seed for reproducibility
np.random.seed(42)

# Load the master CSV file
data = pd.read_csv('../data/ZTFBTS/ZTFBTS_TransientTable_withpeak.csv')

# Shuffle the rows randomly
data_shuffled = data.sample(frac=1, random_state=42)  # Using a fixed random_state for reproducibility

# Calculate the number of rows for training and testing
total_rows = len(data_shuffled)
train_rows = int(total_rows * 4 / 5)
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

debug = False
spec_data = specdata(master_list = "../data/ZTFBTS/ZTFBTS_TransientTable_withpeak_train.csv",
                     max_length = 220, photometry_len = 20,
                     spectra = "../data/ZTFBTS_spectra_minimal",
                     post_fix = "_minimal_AVcorr_zcorr_logF",
                     verbose = True,
                     spectime = True, 
                     combine_photometry=True, 
                     centering=False,
                     midfiltsize = 3,
                     debug=debug, debugread = 8,
                     )
flux, wavelength, mask, type, phase, photoflux, phototime, photowavelength, photomask = spec_data.get_data(concat_photometry = True)
#breakpoint()
np.savez("../data/train_data_align_with_simu_minimal.npz", 
         flux=flux, 
         wavelength=wavelength, 
         mask=mask, 
         type=type, 
         phase=phase, photoflux=photoflux, 
         phototime=phototime, photowavelength=photowavelength,
         photomask=photomask,


        flux_mean=spec_data.fluxes_mean,
        flux_std=spec_data.fluxes_std,
        wavelength_mean=spec_data.wavelengths_mean,
        wavelength_std=spec_data.wavelengths_std,

        green_mean = spec_data.green_mean,
        green_std = spec_data.green_std,
        red_mean = spec_data.red_mean,
        red_std = spec_data.red_std,

        combined_mean = spec_data.combined_mean,
        combined_std = spec_data.combined_std,

        green_time_mean = spec_data.green_time_mean,
        green_time_std = spec_data.green_time_std,
        red_time_mean = spec_data.red_time_mean,
        red_time_std = spec_data.red_time_std,

        combined_time_mean = spec_data.combined_time_mean,
        combined_time_std = spec_data.combined_time_std,
        spectime_mean = spec_data.spectime_mean,
        spectime_std = spec_data.spectime_std, 
        ztfid = spec_data.ztfid)
with open('../data/train_class_dict_align_with_simu.json', 'w') as fp:
    json.dump(spec_data.class_encoding, fp)


class_encoding = spec_data.class_encoding

spec_data_test = specdata(master_list = "../data/ZTFBTS/ZTFBTS_TransientTable_withpeak_test.csv",
                     spectra = "../data/ZTFBTS_spectra_minimal",
                     post_fix = "_minimal_AVcorr_zcorr_logF",
                     verbose = True,
                     spectime = True, 
                     max_length = 220, photometry_len = 20,
                     class_encoding = class_encoding, 
                     midfiltsize = 3,
                     z_score=False,
                     debug=debug, debugread = 8,
                     )
flux, wavelength, mask, type, phase, photoflux, phototime, photowavelength,photomask = spec_data_test.get_data(concat_photometry = True)

np.savez("../data/test_data_align_with_simu_minimal.npz", 
         flux=(flux - spec_data.fluxes_mean) / spec_data.fluxes_std, 
         wavelength=(wavelength - spec_data.wavelengths_mean) / spec_data.wavelengths_std, 
         mask=mask, 
         type=type, 
         phase = (phase - spec_data.spectime_mean) / spec_data.spectime_mean, 
         photoflux=(photoflux - spec_data.combined_mean) / spec_data.combined_std, 
         phototime=(phototime - spec_data.combined_time_mean) / spec_data.combined_time_std, 
         photowavelength= photowavelength,
         photomask=photomask,
         
         flux_mean=spec_data.fluxes_mean,
        flux_std=spec_data.fluxes_std,
        wavelength_mean=spec_data.wavelengths_mean,
        wavelength_std=spec_data.wavelengths_std,

        green_mean = spec_data.green_mean,
        green_std = spec_data.green_std,
        red_mean = spec_data.red_mean,
        red_std = spec_data.red_std,

        combined_mean = spec_data.combined_mean,
        combined_std = spec_data.combined_std,

        green_time_mean = spec_data.green_time_mean,
        green_time_std = spec_data.green_time_std,
        red_time_mean = spec_data.red_time_mean,
        red_time_std = spec_data.red_time_std,

        combined_time_mean = spec_data.combined_time_mean,
        combined_time_std = spec_data.combined_time_std,
        spectime_mean = spec_data.spectime_mean,
        spectime_std = spec_data.spectime_std, 
        ztfid = spec_data_test.ztfid)

