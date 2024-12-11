import h5py
import numpy as np
import os
from cosmolopy import magnitudes, cc
import math
import pysynphot as S
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import sys

lambda_min = 3000
lambda_max = 8000
filtersize = 3
centering = sys.argv[1].lower() == "true"
training_prop = 0.8

data_dir = "../data/goldstein"
np.random.seed(42)
# Load LSST ugrizy filters
filters = ['u', 'g', 'r', 'i', 'z', 'y']
filters_np = [np.genfromtxt(f'../data/filters/LSST/LSST_LSST.{ii}.dat') for ii in filters]
bps = [S.ArrayBandpass(tp[:,0], tp[:,1], name=f'LSST {ii}') for ii, tp in zip(filters, filters_np)]
phase_range = [-10, 0, 10, 20, 30]


wavelengths = []
fluxes = []
masks = []
phase = []

times = []
photometric = []
photometric_mask = []
photometric_band = []

training_or_not = []

file_name = []

for filename in os.listdir(data_dir):
    file_path = os.path.join(data_dir, filename)
    
    if os.path.isfile(file_path):
        with h5py.File(file_path, 'r') as f:
            tim = np.array(f['time']) / (24 * 3600) # seconds to days
            lam = magnitudes.nu_lambda(np.array(f['nu'])) # frequency to wavelength
            Fnu = np.array(f['Lnu']) / (4. * math.pi * (10*cc.pc_cm)**2) # to absolute flux, at 10 pc
            total_brightness = np.sum(Fnu, axis=1)
            in_range_lam = np.where(np.logical_and(lam > lambda_min, lam < lambda_max))
            
            #plt.plot(lam_in, Fnu[np.argmax(total_brightness), in_range_lam][0,:])
            #plt.show()
            #plt.savefig('spectrum.png')
            #breakpoint()
            # now make ZTF light curves
            LCs = {i:[] for i in filters}
            LC_masks = {i:[] for i in filters}
            #LC_r = []
            #LC_g = []
            #LC_r_mask = []
            #LC_g_mask = []
            
            for i in range(tim.shape[0]):
                #breakpoint()
                if np.sum(Fnu[i]) == 0:
                    for ii in filters:
                        LCs[ii].append(0)
                        LC_masks[ii].append(0)
                    continue

                F_object = S.ArraySpectrum(lam, Fnu[i] + 1e-20 * (Fnu[i] == 0), fluxunits='fnu')
                for ii, bp in zip(filters, bps):
                    LC_masks[ii].append(1)
                    LC = S.Observation(F_object, bp, force='taper').effstim('abmag')
                    LCs[ii].append(LC)
            
            LC_concat = np.array([LCs[ii] for ii in filters]).flatten()
            LC_concat_mask = np.array([LC_masks[ii] for ii in filters]).flatten()
            LC_concat_band = np.concatenate([np.zeros_like(LCs[ii]) + i for i, ii in enumerate(filters)])

            # a random number to decide whether it is in training
            in_training = np.random.rand() < training_prop
            peak = np.argmax(total_brightness)
            min_Fnu = np.min(Fnu[Fnu != 0])
            min_Fnu = max(min_Fnu, 1e-18)
            for phase_shift in phase_range:
                if peak + phase_shift > len(tim) or peak + phase_shift < 0:
                    continue # skip this phase if out of bounds
                raw_flux =  Fnu[peak + phase_shift, in_range_lam][0,:]
                if max(raw_flux) == 0:
                    continue # too dark, skip
                mask = (raw_flux != 0) * 1
                raw_flux[raw_flux == 0] = min_Fnu
                spectrum = np.log10( raw_flux)
                #breakpoint()
                lam_in = lam[in_range_lam]
                spectrum = medfilt(spectrum, filtersize)
                '''
                plt.plot(lam_in[mask == 1], spectrum[mask == 1]-np.mean(spectrum))
                plt.show()
                plt.savefig('spectrum.png')
                breakpoint()
                '''
                phase.append(phase_shift)
            
                wavelengths.append(lam_in)
                if centering:
                    fluxes.append(spectrum - np.mean(spectrum))
                else:
                    fluxes.append(spectrum)
                #masks.append(np.ones_like(spectrum))
                #breakpoint()
                masks.append(mask)
                times.append(tim)
                photometric.append(LC_concat)
                photometric_mask.append(LC_concat_mask)
                photometric_band.append(LC_concat_band)
                training_or_not.append(in_training)
                file_name.append(filename)
            #breakpoint()

            #print(LC_r)
            # plot light curves
            '''
            plt.figure()
            for ii in filters:
                plt.scatter(tim, LCs[ii], label=ii)

            plt.ylim((-19.5,-16.5))
            plt.gca().invert_yaxis()
            plt.legend()
            plt.xlabel('Time (days)')
            plt.ylabel('Magnitude')
            plt.title('Light curves')
            plt.show()
            plt.savefig('light_curves.png')
            breakpoint()
            '''
            

# some post-hoc centering/normalizing etc.
fluxes = np.array(fluxes)
masks = np.array(masks)


times = np.array(times)
photometric = np.array(photometric)
photometric_mask = np.array(photometric_mask)
photometric_band = np.array(photometric_band)

phase = np.array(phase)
phase_mean = np.mean(phase)
phase_std = np.std(phase)
phase = (phase - phase_mean) / phase_std



# center and normalize
flux_mean = np.mean(fluxes)
flux_std = np.std(fluxes)
fluxes = (fluxes - flux_mean) / flux_std

wavelengths_mean = np.mean(wavelengths)
wavelengths_std = np.std(wavelengths)
wavelengths = (wavelengths - wavelengths_mean) / wavelengths_std

time_mean = np.mean(times)
time_std = np.std(times)
times = (times - time_mean) / time_std

photometric_mean = np.mean(photometric)
photometric_std = np.std(photometric)
photometric = (photometric - photometric_mean) / photometric_std


training_or_not = np.array(training_or_not)
training_idx = np.where(training_or_not)[0]
testing_idx = np.where(np.logical_not( training_or_not))[0]
file_name = np.array(file_name)

# save to file
np.savez(f'../data/goldstein_processed/preprocessed_midfilt_{filtersize}_centering{centering}_LSST_phase.npz', 
        wavelength=wavelengths, 
        flux=fluxes, 
        mask=masks,
        phase=phase, 

        phototime=np.concatenate( [times for _ in filters], 1),#times, 
        photoflux=photometric, 
        photomask=photometric_mask, 
        photowavelength=photometric_band, 
        
        wavelength_mean=wavelengths_mean,
        wavelength_std=wavelengths_std,
        flux_mean=flux_mean,
        flux_std=flux_std,
        phase_mean=phase_mean,
        phase_std=phase_std,

        phototime_mean=time_mean,
        phototime_std=time_std,
        photoflux_mean=photometric_mean,
        photoflux_std=photometric_std,
        training_idx=training_idx,
        testing_idx=testing_idx,
        identity=file_name
        )



