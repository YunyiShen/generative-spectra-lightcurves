import jax
import jax.numpy as np
import flax
import pandas as pd
import os
from tqdm import tqdm



class specdata:
    def __init__(self, 
                 master_list = "../data/ZTFBTS/ZTFBTS_TransientTable_train.csv", 
                 light_curves = "../data/ZTFBTS/light-curves",
                 spectra = "../data/ZTFBTS_spectra",
                 max_length = 1000, photometry_len = 200, verbose = False):
        self.master_list = pd.read_csv(master_list, header = 0)
        self.light_curves = light_curves
        self.spectra = spectra
        self.max_length = max_length
        self.photometry_len = photometry_len
        self.wavelengths = []
        self.fluxes = []
        self.masks = []
        self.len_list = []
        self.class_list = []
        self.class_encoding = {}
        self.green_time = []
        self.green_flux = []
        self.green_mask = []
        self.red_time = []
        self.red_flux = []
        self.red_mask = []
        self.fluxes_mean = 0.
        self.fluxes_std = 0.
        self.wavelengths_mean = 0.
        self.wavelengths_std = 0.

        self.red_mean = 0.
        self.red_std = 0.
        self.green_mean = 0.
        self.green_std = 0.

        self.green_time_mean = 0.
        self.green_time_std = 0.
        self.red_time_mean = 0.
        self.red_time_std = 0.
        self.load_data(verbose = verbose)

    def load_data(self, verbose = False):
        for _, row in tqdm(self.master_list.iterrows(), total=len(self.master_list)):
            if not os.path.exists(f"{self.spectra}/{row['ZTFID']}.csv"):
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its spectum doesn't exist")
                continue
            if not os.path.exists(f"{self.light_curves}/{row['ZTFID']}.csv"):
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its light curve doesn't exist")
                continue



            spectra_pd = pd.read_csv(f"{self.spectra}/{row['ZTFID']}.csv", header=None)
            photometry_pd = pd.read_csv(f"{self.light_curves}/{row['ZTFID']}.csv", header=0)
            green = photometry_pd[photometry_pd['band'] == 'g'][['time','mag']]
            red = photometry_pd[photometry_pd['band'] == 'R'][['time','mag']]
            red_shift = row['redshift'] # redshift

            self.len_list.append(len(spectra_pd))
    
            if len(spectra_pd) > self.max_length:
                if verbose:
                    print(f"Skipping {row['ZTFID']} because it's too long with length {len(spectra_pd)}")
                continue
            if len(green) > self.photometry_len: 
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its green band too long with length {len(green)}")
                continue
            if len(red) > self.photometry_len: 
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its red band too long with length {len(red)}")
                continue
            ### type
            self.class_list.append(row['type']) # type
            
            ### spectra
            wavelength = spectra_pd[0].values * (1. + float(red_shift)) # correct for redshift
            flux = spectra_pd[1].values
    
            # Pad the flux, wavelength, and mask
            flux = np.pad(flux, (0, self.max_length - len(flux)))
            wavelength = np.pad(wavelength, (0, self.max_length - len(wavelength)))
    
            # Mask should be zero where the padding is
            mask = np.ones(self.max_length)
            mask = mask.at[len(spectra_pd):].set(0)

            # Append to the list
            self.fluxes.append(flux)
            self.wavelengths.append(wavelength)
            self.masks.append(mask)

            ### photometry
            # green band
            green_time = green['time'].values
            green_flux = green['mag'].values
            green_time = np.pad(green_time, (0, self.photometry_len - len(green_time)))
            green_flux = np.pad(green_flux, (0, self.photometry_len - len(green_flux)))

            green_mask = np.ones(self.photometry_len)
            green_mask = green_mask.at[len(green):].set(0)

            self.green_time.append(green_time)
            self.green_flux.append(green_flux)
            self.green_mask.append(green_mask)

            # red band
            red_time = red['time'].values
            red_flux = red['mag'].values
            red_time = np.pad(red_time, (0, self.photometry_len - len(red_time)))
            red_flux = np.pad(red_flux, (0, self.photometry_len - len(red_flux)))

            red_mask = np.ones(self.photometry_len)
            red_mask = red_mask.at[len(red):].set(0)

            self.red_time.append(red_time)
            self.red_flux.append(red_flux)
            self.red_mask.append(red_mask)

        # Stack the fluxes, spectra, and masks  
        self.fluxes = np.stack(self.fluxes)
        self.wavelengths = np.stack(self.wavelengths)
        self.masks = np.stack(self.masks)
        self.green_time = np.stack(self.green_time)
        self.green_flux = np.stack(self.green_flux)
        self.green_mask = np.stack(self.green_mask)
        self.red_time = np.stack(self.red_time)
        self.red_flux = np.stack(self.red_flux)
        self.red_mask = np.stack(self.red_mask)

    
        # Convert masks to bool
        self.masks = self.masks.astype(bool)
        self.green_mask = self.green_mask.astype(bool)
        self.red_mask = self.red_mask.astype(bool)

        # Normalize the fluxes and wavelengths
        self.fluxes_mean = np.mean(self.fluxes)
        self.fluxes_std = np.std(self.fluxes)

        self.wavelengths_mean = np.mean(self.wavelengths)
        self.wavelengths_std = np.std(self.wavelengths)

        self.fluxes = (self.fluxes - self.fluxes_mean) / self.fluxes_std
        self.wavelengths = (self.wavelengths - self.wavelengths_mean) / self.wavelengths_std

        # Normalize photometry and time
        self.green_mean = np.mean(self.green_flux)
        self.green_std = np.std(self.green_flux)
        self.red_mean = np.mean(self.red_flux)
        self.red_std = np.std(self.red_flux)

        self.green_time_mean = np.mean(self.green_time)
        self.green_time_std = np.std(self.green_time)
        self.red_time_mean = np.mean(self.red_time)
        self.red_time_std = np.std(self.red_time)

        self.green_flux = (self.green_flux - self.green_mean) / self.green_std
        self.red_flux = (self.red_flux - self.red_mean) / self.red_std
        self.green_time = (self.green_time - self.green_time_mean) / self.green_time_std
        self.red_time = (self.red_time - self.red_time_mean) / self.red_time_std

        # encode classes
        
        self.class_encoding = {cls: idx for idx, cls in enumerate(set(self.class_list))}
        self.class_list = [self.class_encoding[cls] for cls in self.class_list]
            

    def get_data(self):
        return self.fluxes, self.wavelengths, self.masks, self.class_list, self.green_flux, self.green_time, self.green_mask, self.red_flux, self.red_time, self.red_mask