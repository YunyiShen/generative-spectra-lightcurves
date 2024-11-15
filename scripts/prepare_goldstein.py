import h5py
import numpy as np
import os
from cosmolopy import magnitudes, cc
import math
import pysynphot as S
import matplotlib.pyplot as plt
from scipy.signal import medfilt

lambda_min = 3000
lambda_max = 8000
filtersize = 3

data_dir = "../data/goldstein"

# Load ZTF filters
tp_ZTF_g = np.genfromtxt('../data/Palomar_ZTF.g.dat')
tp_ZTF_r = np.genfromtxt('../data/Palomar_ZTF.r.dat')
bp_g = S.ArrayBandpass(tp_ZTF_g[:,0], tp_ZTF_g[:,1], name='ZTF g')
bp_r = S.ArrayBandpass(tp_ZTF_r[:,0], tp_ZTF_r[:,1], name='ZTF r')
wavelengths = []
fluxes = []
masks = []
phase = []

times = []
photometric = []
photometric_mask = []
photometric_band = []

for filename in os.listdir(data_dir):
    file_path = os.path.join(data_dir, filename)
    
    if os.path.isfile(file_path):
        with h5py.File(file_path, 'r') as f:
            tim = np.array(f['time']) / (24 * 3600) # seconds to days
            lam = magnitudes.nu_lambda(np.array(f['nu'])) # frequency to wavelength
            Fnu = np.array(f['Lnu']) / (4. * math.pi * (10*cc.pc_cm)**2) # to absolute flux, at 10 pc
            total_brightness = np.sum(Fnu, axis=1)
            in_range_lam = np.where(np.logical_and(lam > lambda_min, lam < lambda_max))
            spectrum = np.log10( Fnu[np.argmax(total_brightness), in_range_lam])[0,:]
            lam_in = lam[in_range_lam]
            spectrum = medfilt(spectrum, filtersize)
            #plt.plot(lam_in, Fnu[np.argmax(total_brightness), in_range_lam][0,:])
            #plt.show()
            #plt.savefig('spectrum.png')
            #breakpoint()
            # now make ZTF light curves
            LC_r = []
            LC_g = []
            LC_r_mask = []
            LC_g_mask = []
            
            for i in range(tim.shape[0]):
                #breakpoint()
                if np.sum(Fnu[i]) == 0:
                    LC_g.append(0)
                    LC_r.append(0)
                    LC_g_mask.append(0)
                    LC_r_mask.append(0)
                    continue
                LC_g_mask.append(1)
                LC_r_mask.append(1)
                F_object = S.ArraySpectrum(lam, Fnu[i] + 1e-20 * (Fnu[i] == 0), fluxunits='fnu')
                #breakpoint()
                ZTF_g = S.Observation(F_object, bp_g, force='taper').effstim('abmag')
                ZTF_r = S.Observation(F_object, bp_r, force='taper').effstim('abmag')
                LC_r.append(ZTF_r)
                LC_g.append(ZTF_g)
            phase.append(0)
            LC_concat = np.array(LC_r + LC_g)
            #LC_r = np.array(LC_r)
            #LC_g = np.array(LC_g)
            
            LC_concat_mask = np.array(LC_r_mask + LC_g_mask)
            #LC_r_mask = np.array(LC_r_mask)
            #LC_g_mask = np.array(LC_g_mask)
            
            LC_concat_band = np.concatenate((np.zeros_like(LC_r), np.ones_like(LC_g)))

            wavelengths.append(lam_in)
            fluxes.append(spectrum - np.mean(spectrum))
            masks.append(np.ones_like(spectrum))
            times.append(tim)
            photometric.append(LC_concat)
            photometric_mask.append(LC_concat_mask)
            photometric_band.append(LC_concat_band)
            #breakpoint()

            #print(LC_r)
            # plot light curves
            '''
            plt.figure()
            plt.scatter(tim, LC_r, label='r')
            plt.scatter(tim, LC_g, label='g')
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

np.random.seed(42)
shuffle = np.random.permutation(len(fluxes))
training_idx = shuffle[:int(0.8 * len(fluxes))]
testing_idx = shuffle[int(0.8 * len(fluxes)):]

# save to file
np.savez(f'../data/goldstein_processed/preprocessed_midfilt_{filtersize}.npz', 
        wavelength=wavelengths, 
        flux=fluxes, 
        mask=masks,
        phase=phase, 

        phototime=np.concatenate( (times, times), 1),#times, 
        photoflux=photometric, 
        photomask=photometric_mask, 
        photowavelength=photometric_band, 
        
        wavelength_mean=wavelengths_mean,
        wavelength_std=wavelengths_std,
        flux_mean=flux_mean,
        flux_std=flux_std,
        phototime_mean=time_mean,
        phototime_std=time_std,
        photoflux_mean=photometric_mean,
        photoflux_std=photometric_std,
        training_idx=training_idx,
        testing_idx=testing_idx
        )



