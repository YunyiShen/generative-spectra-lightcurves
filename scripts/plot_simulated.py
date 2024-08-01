import h5py
import numpy as np

import matplotlib.pyplot as plt

def normalize_spectrum(spectra):
    return (spectra - np.min(spectra,axis = 1)[:,None]) / (np.max(spectra,axis=1) - np.min(spectra,axis = 1))[:,None]   


file = h5py.File("../data/ZTF_Pretrain_5Class_ZFLAT_PERFECT.hdf5", 'r')

snss = ['SN II', 'SN Ia', 'SN Ib', 'SN Ic']

#sns = "SN Ib"
for sns in snss:
    keyy = [key for key in file['Photometry'][sns].keys()][0]
    spect_tmp = np.array(file['Spectroscopy'][sns][keyy]['flux_perfect'])
    spec_mask_tmp = spect_tmp != -999.

    spect_tmp = spect_tmp * spec_mask_tmp
    spect_tmp = normalize_spectrum(spect_tmp)

    zs = np.array(file['Spectroscopy'][sns][keyy]['z'])
    wavelength_tmp = np.array(file['Spectroscopy'][sns][keyy]['wavelength'])/(1 + zs[:,None]) # redshift correction
    wavelength_tmp *= spec_mask_tmp
    #breakpoint()
    np.save(f"{sns}_simulated.npy", spect_tmp[:100])
    np.save(f"{sns}_wavelength.npy", wavelength_tmp[:100])
