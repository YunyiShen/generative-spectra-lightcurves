import jax
import jax.numpy as np
import flax
import pandas as pd
import os
from tqdm import tqdm
from models.gp import gp_peak_time



class specdata:
    def __init__(self, 
                 master_list = "../data/ZTFBTS/ZTFBTS_TransientTable_withpeak.csv", 
                 light_curves = "../data/ZTFBTS/light-curves",
                 spectra = "../data/ZTFBTS_spectra_dashed",
                 max_length = 1024, photometry_len = 256, 
                 class_encoding = None,
                 z_score = True, 
                 spectime = False, # this is relative to peak
                 band_for_peak = "r",
                 min_length_for_peak = 5,
                 verbose = False):
        self.master_list = pd.read_csv(master_list, header = 0)
        self.light_curves = light_curves
        self.spectra = spectra
        self.max_length = max_length
        self.photometry_len = photometry_len
        self.min_length_for_peak = min_length_for_peak
        self.wavelengths = []
        self.fluxes = []
        self.masks = []
        self.len_list = []
        self.class_list = []
        self.class_encoding = class_encoding
        self.green_time = []
        self.green_flux = []
        self.green_wavelength = []
        self.green_mask = []
        self.red_time = []
        self.red_flux = []
        self.red_wavelength = []
        self.red_mask = []
        self.fluxes_mean = 0.
        self.fluxes_std = 1.
        self.wavelengths_mean = 0.
        self.wavelengths_std = 1.

        self.red_mean = 0.
        self.red_std = 1.
        self.green_mean = 0.
        self.green_std = 1.

        self.green_time_mean = 0.
        self.green_time_std = 1.
        self.red_time_mean = 0.
        self.red_time_std = 1.
        
        self.spectime = spectime
        self.spectime_list = []
        self.spectime_mean = 0.
        self.spectime_std = 1.

        self.peak_band = band_for_peak

        self.peak_time = []

        self.load_data(verbose = verbose)
        self.num_class = len(self.class_encoding.keys())
        self.z_score = z_score
        if z_score:
            self.normalize()


    def load_data(self, verbose = False):
        for _, row in tqdm(self.master_list.iterrows(), total=len(self.master_list)):
            if not os.path.exists(f"{self.spectra}/{row['ZTFID']}_dashed.csv"):
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its spectrum doesn't exist")
                continue
            if not os.path.exists(f"{self.light_curves}/{row['ZTFID']}.csv"):
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its light curve doesn't exist")
                continue


            try:
                spectra_pd = pd.read_csv(f"{self.spectra}/{row['ZTFID']}_dashed.csv", header=None)
                photometry_pd = pd.read_csv(f"{self.light_curves}/{row['ZTFID']}.csv", header=0)
                green = photometry_pd[photometry_pd['band'] == 'g'][['time','mag']]
                red = photometry_pd[photometry_pd['band'] == 'R'][['time','mag']]
            except:
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its spectrum or light curve is corrupted")
                continue
            

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

            if len(red) < self.min_length_for_peak and self.peak_band == "r":
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its red band too short for peak determination with length {len(red)}")
                continue
            if len(green) < self.min_length_for_peak and self.peak_band == "g":
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its green band too short for peak determination with length {len(green)}")
                continue
            ### type
            if self.spectime:
                spectime = row['Spectrum_Date_(MJD)']
                #self.spectime_list.append(spectime)
            #breakpoint()
            ### spectra
            wavelength = spectra_pd[0].values #* (0. if np.isnan( float(red_shift)) else float(red_shift) + 1.) # correct for redshift
            if np.max(wavelength) <= 2000. or np.min(wavelength) >= 8000.:
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its spectrum is out of range")
                continue
            
            flux = spectra_pd[1].values
            flux = flux[np.argsort(wavelength)]
            wavelength = wavelength[np.argsort(wavelength)]
    
            # Pad the flux, wavelength, and mask
            flux = np.pad(flux, (0, self.max_length - len(flux)))
            wavelength = np.pad(wavelength, (0, self.max_length - len(wavelength)))
    
            # Mask should be zero where the padding is
            mask = np.ones(self.max_length)
            mask = mask.at[len(spectra_pd):].set(0)

            

            ### photometry
            # green band
            green_time = green['time'].values
            green_flux = green['mag'].values
            green_wavelength = 0.*green_time + 1196.25

            if self.peak_band not in ["g", "r"]:
                raise ValueError("band for calculating peak must be either 'g' (green) or 'r' (red)")

            if self.spectime and self.peak_band == "g":
                peak = gp_peak_time(green_time, green_flux)
                spectime -= peak
                if spectime >= 200: # 200 days old
                    if verbose:
                        print(f"Skipping {row['ZTFID']} because spectrum is taken way far from peak")
                    continue
                self.peak_time.append(peak)

            green_time = np.pad(green_time, (0, self.photometry_len - len(green_time)))
            green_flux = np.pad(green_flux, (0, self.photometry_len - len(green_flux)))
            green_wavelength = np.pad(green_wavelength, (0, self.photometry_len - len(green_wavelength)))

            green_mask = np.ones(self.photometry_len)
            green_mask = green_mask.at[len(green):].set(0)

            

            # red band
            red_time = red['time'].values
            red_flux = red['mag'].values
            red_wavelength = 0.*red_time + 6366.38

            #print(row['ZTFID'])
            if self.spectime and self.peak_band == "r":
                peak = gp_peak_time(red_time, red_flux)
                spectime -= peak
                if spectime >= 200: # 200 days old
                    if verbose:
                        print(f"Skipping {row['ZTFID']} because spectrum is taken way far from peak")
                    continue
                self.peak_time.append(peak)
            

            red_time = np.pad(red_time, (0, self.photometry_len - len(red_time)))
            red_flux = np.pad(red_flux, (0, self.photometry_len - len(red_flux)))
            red_wavelength = np.pad(red_wavelength, (0, self.photometry_len - len(red_wavelength)))

            red_mask = np.ones(self.photometry_len)
            red_mask = red_mask.at[len(red):].set(0)



            # Append to the list
            if self.class_encoding is not None:
                if row['type'] not in self.class_encoding:
                    if verbose:
                        print(f"Skipping {row['ZTFID']} because its class is not not in training")
                    continue

            self.class_list.append(row['type']) # type
            if self.spectime:
                self.spectime_list.append(spectime)

            # flux
            self.fluxes.append(flux)
            self.wavelengths.append(wavelength)
            self.masks.append(mask)
            
            self.green_time.append(green_time)
            self.green_flux.append(green_flux)
            self.green_wavelength.append(green_wavelength)
            self.green_mask.append(green_mask)

            self.red_time.append(red_time)
            self.red_flux.append(red_flux)
            self.red_wavelength.append(red_wavelength)
            self.red_mask.append(red_mask)

        # Stack the fluxes, spectra, and masks  
        self.fluxes = np.stack(self.fluxes)
        self.wavelengths = np.stack(self.wavelengths)
        self.masks = np.stack(self.masks)
        self.green_time = np.stack(self.green_time)
        self.green_flux = np.stack(self.green_flux)
        self.green_wavelength = np.stack(self.green_wavelength)
        self.green_mask = np.stack(self.green_mask)
        self.red_time = np.stack(self.red_time)
        self.red_flux = np.stack(self.red_flux)
        self.red_wavelength = np.stack(self.red_wavelength)
        self.red_mask = np.stack(self.red_mask)

    
        # Convert masks to bool
        self.masks = self.masks.astype(bool)
        self.green_mask = self.green_mask.astype(bool)
        self.red_mask = self.red_mask.astype(bool)

        # encode classes
        if self.class_encoding is None:
            self.class_encoding = {cls: idx for idx, cls in enumerate(set(self.class_list))}
        self.class_list = np.array([self.class_encoding[cls] for cls in self.class_list])

        if self.spectime:
            self.spectime_list = np.array(self.spectime_list)
            

    def get_data(self, concat_photometry = False):
        if concat_photometry:
            if self.spectime:
                return self.fluxes, self.wavelengths, self.masks, self.class_list, self.spectime_list, np.concatenate([self.green_flux, self.red_flux], axis = 1), np.concatenate([self.green_time, self.red_time], axis = 1), np.concatenate([self.green_wavelength, self.red_wavelength], axis = 1), np.concatenate([self.green_mask, self.red_mask], axis = 1)
            return self.fluxes, self.wavelengths, self.masks, self.class_list, np.concatenate([self.green_flux, self.red_flux], axis = 1), np.concatenate([self.green_time, self.red_time], axis = 1),np.concatenate([self.green_wavelength, self.red_wavelength], axis = 1), np.concatenate([self.green_mask, self.red_mask], axis = 1)
        else:
            if self.spectime:
                return self.fluxes, self.wavelengths, self.masks, self.class_list, self.spectime_list, self.green_flux, self.green_time, self.green_wavelength,self.green_mask, self.red_flux, self.red_time, self.red_wavelength,self.red_mask
            return self.fluxes, self.wavelengths, self.masks, self.class_list, self.green_flux, self.green_time, self.green_wavelength, self.green_mask, self.red_flux, self.red_time, self.red_wavelength,self.red_mask
    
    def normalize(self):
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

        self.green_wavelength = (self.green_wavelength - self.wavelengths_mean)/self.wavelengths_std
        self.red_wavelength = (self.red_wavelength - self.wavelengths_mean)/self.wavelengths_std


        if self.spectime:
            self.spectime_mean = np.mean(self.spectime_list)
            self.spectime_std = np.std(self.spectime_list)
            self.spectime_list = (self.spectime_list - self.spectime_mean) / self.spectime_std