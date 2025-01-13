import jax
import jax.numpy as np
import numpy as onp
import flax
import pandas as pd
import os
from tqdm import tqdm
from models.gp import gp_peak_time
from scipy.signal import medfilt

def normalizing_spectra(spectra):
    # normalizing to 0-1 range for each spectrum
    min_val = np.min(spectra, axis = 1, keepdims = True)
    max_val = np.max(spectra, axis = 1, keepdims = True)
    return (spectra - min_val)/(max_val - min_val)

def random_subsample(an_array, maxlen):
    if len(an_array) > maxlen:
        indices = onp.random.choice(len(an_array), size=maxlen, replace=False)
        indices.sort()
        #breakpoint()
        return an_array.iloc[indices]
    else:
        return an_array


class specdata:
    def __init__(self, 
                 master_list = "../data/ZTFBTS/ZTFBTS_TransientTable_withpeak.csv", 
                 light_curves = "../data/ZTFBTS/light-curves",
                 spectra = "../data/ZTFBTS_spectra",
                 post_fix = "",
                 max_length = 1024, photometry_len = 256, 
                 class_encoding = None,
                 z_score = True, 
                 spectime = False, # this is relative to peak
                 band_for_peak = "r",
                 min_length_for_peak = 5,
                 phase_cutoff = [-20, 70],
                 combine_photometry = True,
                 centering = False,

                 midfiltsize = 3,
                 onlyIa = False,
                 verbose = False,
                 debug = False, debugread = 8):
        self.master_list = pd.read_csv(master_list, header = 0)
        self.light_curves = light_curves
        self.spectra = spectra
        self.onlyIa = onlyIa
        self.midfiltsize = midfiltsize
        self.max_length = max_length
        self.photometry_len = photometry_len
        self.min_length_for_peak = min_length_for_peak
        self.phase_cutoff = phase_cutoff
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
        self.ztfid = []
        self.post_fix = post_fix  
        self.centering = centering

        self.load_data(verbose = verbose, debug = debug, debugread = debugread)
        self.num_class = len(self.class_encoding.keys())
        self.z_score = z_score
        self.combine_photometry = combine_photometry
        if z_score:
            self.normalize(combine_photometry)


    def load_data(self, verbose = False, 
                 debug = False, debugread = 8):
        for ii, row in tqdm(self.master_list.iterrows(), total=len(self.master_list)):
            if debug and ii == debugread:
                break
            
            if not os.path.exists(f"{self.spectra}/{row['ZTFID']}{self.post_fix}.csv"):
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its spectrum doesn't exist")
                continue
            if not os.path.exists(f"{self.light_curves}/{row['ZTFID']}.csv"):
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its light curve doesn't exist")
                continue
            if self.onlyIa and "Ia" not in row['type']:
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its type is not Ia")
                continue

            try:
                spectra_pd = pd.read_csv(f"{self.spectra}/{row['ZTFID']}{self.post_fix}.csv", header=None)
                photometry_pd = pd.read_csv(f"{self.light_curves}/{row['ZTFID']}.csv", header=0)
                green = photometry_pd[photometry_pd['band'] == 'g'][['time','mag']]
                red = photometry_pd[photometry_pd['band'] == 'R'][['time','mag']]
            except:
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its spectrum or light curve is corrupted")
                continue
            

            
            if len(red) < self.min_length_for_peak and self.peak_band == "r":
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its red band too short for peak determination with length {len(red)}")
                continue
            if len(green) < self.min_length_for_peak and self.peak_band == "g":
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its green band too short for peak determination with length {len(green)}")
                continue

            ### photometry
            # green band
            green_time = green['time'].values
            green_flux = green['mag'].values
            green_wavelength = np.zeros_like(green_time) #0.*green_time + 1196.25

            if self.peak_band not in ["g", "r"]:
                raise ValueError("band for calculating peak must be either 'g' (green) or 'r' (red)")

            if self.spectime and self.peak_band == "g":
                peak = gp_peak_time(green_time, green_flux)


            # red band
            red_time = red['time'].values
            red_flux = red['mag'].values
            red_wavelength = np.zeros_like(red_time) + 1 #0.*red_time + 6366.38
            #breakpoint()
            #print(row['ZTFID'])
            if self.spectime and self.peak_band == "r":
                peak = gp_peak_time(red_time, red_flux)
            
            
            green_time -= peak # relative to peak
            red_time -= peak

            if np.all(green_time <= .1) or np.all(green_time >= -.1):
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its green band does not cover peak")
                continue
            if np.all(red_time <= .1) or np.all(red_time >= -.1):
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its red band does not cover peak")
                continue

            time_keep = np.logical_and(green_time >= self.phase_cutoff[0], green_time <= self.phase_cutoff[1])

            if np.sum(time_keep) < 3:
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its green band does not have enough measurement")
                continue
            if np.sum(time_keep) > self.photometry_len:
                if verbose:
                    print(f"subsampling {row['ZTFID']} because its green band has too many measurements")
                #breakpoint()
                time_keep = onp.random.choice(np.where(time_keep)[0], size = self.photometry_len, replace = False)
                time_keep.sort()
            else:
                time_keep = np.where(time_keep)[0]

            green_time = green_time[time_keep]
            green_time = np.pad(green_time, (0, self.photometry_len - len(green_time)))
            green_flux = green_flux[time_keep]
            green_flux = np.pad(green_flux, (0, self.photometry_len - len(green_flux)))
            green_wavelength = green_wavelength[time_keep]
            green_wavelength = np.pad(green_wavelength, (0, self.photometry_len - len(green_wavelength)))

            green_mask = np.ones(self.photometry_len)
            green_mask = green_mask.at[len(time_keep):].set(0)
            

            #### red ###
            time_keep = np.logical_and(red_time >= self.phase_cutoff[0], red_time <= self.phase_cutoff[1])
            
            if np.sum(time_keep) < 3:
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its red band does not have enough measurement")
                continue
            if np.sum(time_keep) > self.photometry_len:
                if verbose:
                    print(f"subsampling {row['ZTFID']} because its red band has too many measurements")
                time_keep = onp.random.choice(np.where(time_keep)[0], size = self.photometry_len, replace = False)
                time_keep.sort()
            else:
                time_keep = np.where(time_keep)[0]

            red_time = red_time[time_keep]
            red_time = np.pad(red_time, (0, self.photometry_len - len(red_time)))
            red_flux = red_flux[time_keep]
            red_flux = np.pad(red_flux, (0, self.photometry_len - len(red_flux)))
            red_wavelength = red_wavelength[time_keep]
            red_wavelength = np.pad(red_wavelength, (0, self.photometry_len - len(red_wavelength)))

            red_mask = np.ones(self.photometry_len)
            red_mask = red_mask.at[len(time_keep):].set(0)

            
            ### type
            if self.spectime:
                spectime = row['Spectrum_Date_(MJD)']
                spectime -= peak
                if np.abs(spectime) >= 100: # 200 days old
                    if verbose:
                        print(f"Skipping {row['ZTFID']} because spectrum is taken way far from peak")
                    continue
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

            keep = np.logical_and(np.isfinite(flux), wavelength >= 2000.)
            keep = np.logical_and(keep, wavelength <= 8000.)

            if np.sum(keep) < 20:
                if verbose:
                    print(f"Skipping {row['ZTFID']} because its spectrum does not have enough measurement")
                continue
            if np.sum(keep) > self.max_length:
                if verbose:
                    print(f"subsampling {row['ZTFID']} because its spectrum has too many measurements")
                keep = onp.random.choice(np.where(keep)[0], size = self.max_length, replace = False)
                keep.sort()
            flux = flux[keep]
            wavelength = wavelength[keep]
            flux = medfilt(flux, self.midfiltsize)
            mask = np.ones(self.max_length)
            mask = mask.at[len(wavelength):].set(0)
            if self.centering:
                flux = flux - np.mean(flux)

            # Pad the flux, wavelength, and mask
            flux = np.pad(flux, (0, self.max_length - len(flux)))
            wavelength = np.pad(wavelength, (0, self.max_length - len(wavelength)))

            # Mask should be zero where the padding is
            
            #breakpoint()
            

            
            #breakpoint()


            # Append to the list
            if self.class_encoding is not None:
                if row['type'] not in self.class_encoding:
                    if verbose:
                        print(f"Skipping {row['ZTFID']} because its class is not not in training")
                    continue

            self.class_list.append(row['type']) # type
            if self.spectime:
                self.spectime_list.append(spectime)

            self.len_list.append(len(spectra_pd))
            self.peak_time.append(peak)
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
            self.ztfid.append(row['ZTFID'])
            #breakpoint()
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
        #self.ztfid = np.array(self.ztfid)

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
    
    def normalize(self, combine_photometry = False):
        # Normalize the fluxes and wavelengths
        self.fluxes_mean = np.mean(self.fluxes[self.masks])
        self.fluxes_std = np.std(self.fluxes[self.masks])

        self.wavelengths_mean = np.mean(self.wavelengths[self.masks])
        self.wavelengths_std = np.std(self.wavelengths[self.masks])

        self.fluxes = (self.fluxes - self.fluxes_mean) / self.fluxes_std
        self.fluxes = self.fluxes * self.masks
        self.wavelengths = (self.wavelengths - self.wavelengths_mean) / self.wavelengths_std
        self.wavelengths = self.wavelengths * self.masks

        # Normalize photometry and time
        self.green_mean = np.mean(self.green_flux[self.green_mask])
        self.green_std = np.std(self.green_flux[self.green_mask])
        self.red_mean = np.mean(self.red_flux[self.red_mask])
        self.red_std = np.std(self.red_flux[self.red_mask])

        self.combined_mean = np.mean(np.concatenate([self.green_flux[self.green_mask].flatten(), self.red_flux[self.red_mask].flatten()], axis = 0))
        self.combined_std = np.std(np.concatenate([self.green_flux[self.green_mask].flatten(), self.red_flux[self.red_mask].flatten()]), axis = 0)


        self.green_time_mean = np.mean(self.green_time[self.green_mask])
        self.green_time_std = np.std(self.green_time[self.green_mask])
        self.red_time_mean = np.mean(self.red_time[self.red_mask])
        self.red_time_std = np.std(self.red_time[self.red_mask])

        self.combined_time_mean = np.mean(np.concatenate([self.green_time[self.green_mask].flatten(), self.red_time[self.red_mask].flatten()], axis = 0))
        self.combined_time_std = np.std(np.concatenate([self.green_time[self.green_mask].flatten(), self.red_time[self.red_mask].flatten()], axis = 0))
        
        if combine_photometry:
            self.green_flux = (self.green_flux - self.combined_mean) / self.combined_std
            self.green_flux = self.green_flux * self.green_mask
            self.red_flux = (self.red_flux - self.combined_mean) / self.combined_std
            self.red_flux = self.red_flux * self.red_mask
            self.green_time = (self.green_time - self.combined_time_mean) / self.combined_time_std
            self.green_time = self.green_time * self.green_mask
            self.red_time = (self.red_time - self.combined_time_mean) / self.combined_time_std # peak at 0
            self.red_time = self.red_time * self.red_mask
        else:
            self.green_flux = (self.green_flux - self.green_mean) / self.green_std
            self.green_flux = self.green_flux * self.green_mask
            self.red_flux = (self.red_flux - self.red_mean) / self.red_std
            self.red_flux = self.red_flux * self.red_mask
            self.green_time = (self.green_time - self.green_time_mean) / self.green_time_std
            self.green_time = self.green_time * self.green_mask
            self.red_time = (self.red_time - self.red_time_mean) / self.red_time_std
            self.red_time = self.red_time * self.red_mask
        #self.green_wavelength = (self.green_wavelength - self.wavelengths_mean)/self.wavelengths_std
        #self.red_wavelength = (self.red_wavelength - self.wavelengths_mean)/self.wavelengths_std
        #breakpoint()

        if self.spectime:
            self.spectime_mean = np.mean(self.spectime_list)
            self.spectime_std = np.std(self.spectime_list)
            self.spectime_list = (self.spectime_list - self.spectime_mean) / self.spectime_std