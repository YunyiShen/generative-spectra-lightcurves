import numpy as np
import pandas as pd
import sys
import os
import astropy.units as u
from astropy.coordinates import SkyCoord
import rubin_sim.maf.db as db
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import sqlite3


def simulate_lsstLC(sed_surface, spec_time, spec_wavelengths, 
                    LSSTschedule = "./data/cadence/baseline_v3.3_10yrs.db",
                    filters_encoding = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5},
                    #filters = ["u", "g", "r", "i", "z", "y"], 
                    filters_loc = "./data/filters/LSST", 
                    len_per_filter = 50,
                    minimum_LC_size = 10,
                    max_retry_location = 500):
    filters = filters_encoding.keys()
    h = 6.626e-27  # Planck's constant (erg·s)
    c = 2.998e10*u.cm/u.s   # Speed of light (cm/s)
    c_AAs = (c).to(u.AA/u.s).value
    
    LSST_rad = 321.15  # Radius in cm
    LSST_area = np.pi * LSST_rad**2  # Collecting area in cm²
    exp_time = 30.0  # Exposure time in seconds
    pix_scale = 0.2  # Pixel scale in arcsec/pixel
    readout_noise = 12.7  # Internal noise in electrons/pixel
    gain = 2.2  # Gain in electrons/photon
    fov = 3.5 #degrees on each side

    tmax_d = 200 # max phase of the transient in days
    tmax_s = tmax_d*86400

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    dist_cm = cosmo.luminosity_distance(z=0).to(u.cm).value
    # Connect to server
    cnx = sqlite3.connect(LSSTschedule)

    # Get a cursor
    cur = cnx.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    df = pd.read_sql_query('SELECT fieldRA, fieldDec, seeingFwhmEff, observationStartMJD, filter, fiveSigmaDepth, skyBrightness, proposalId  FROM observations', cnx)
    cnx.close()
    # Generate num_points random coordinates
    num_events = 1

    #if randomly generating locations, uncomment below
    for _ in range(max_retry_location):
        eventRA = np.random.uniform(0, 2 * np.pi, num_events) * u.rad
        eventRA = eventRA.to(u.deg).value
        eventDec = np.arcsin(np.random.uniform(-1, 1, num_events)) * u.rad
        eventDec = eventDec.to(u.deg).value


        dRA = np.abs(df['fieldRA'].values - eventRA)
        dRA = np.minimum(dRA, 360 - dRA)
        dDec = np.abs(df['fieldDec'] - eventDec)

        df_obs = df[(dRA < fov/2) &
            (dDec < fov/2)]

        df_obs.dropna(inplace=True)

        tmin_obs = np.min(df_obs['observationStartMJD'])
        tmax_obs = np.max(df_obs['observationStartMJD'])

        # Pick a random start time for the transient
        try:
            start_mjd = np.random.uniform(tmin_obs, tmax_obs/2)
        except:
            start_mjd = tmin_obs

        shifted_times = spec_time/86400 + start_mjd
        # Exclude points beyond max_phase
        df_obs = df_obs[(df_obs['observationStartMJD'] >= start_mjd) & (df_obs['observationStartMJD'] <= (start_mjd+tmax_d))]
        df_obs.sort_values(by=['observationStartMJD'], inplace=True)
        if len(df_obs) > 0:
        #resample the SED surface to the times of the observations
            new_s = (df_obs['observationStartMJD'].values - start_mjd)*86400
            sed_surface = interp1d(spec_time, sed_surface, axis=0, bounds_error=False, fill_value=0)(new_s)
            spec_time = new_s

            # mjd dates of the observations
            mjd = df_obs['observationStartMJD'].values

            sky_brightness_mags = df_obs['skyBrightness'].values  # Sky brightness, in mag/arcsec^2
            fwhm_eff = df_obs['seeingFwhmEff'].values  # Seeing FWHM in arcsec
            filters = df_obs['filter'].values  # Filters for each observation

            # Effective number of pixels for a point source (based on the nightly seeing and pixel scale)
            n_eff = 2.266 * (fwhm_eff / pix_scale)**2

            bpass_dict = {}
            for flt in filters:
                band = pd.read_csv(f"{filters_loc}/LSST_LSST.{flt}.dat", header=None, sep='\s+', names=['w','t'])
                bpass_dict[flt] = S.ArrayBandpass(band['w'].values, band['t'].values)

            # Compute Signal, Background, and Magnitudes for each observation
            filts = df_obs['filter'].values
            bandpasses = np.array([bpass_dict[filt] for filt in filts])

            source_mag = []
            # Loop through each observation
            for i, bandpass in enumerate(bandpasses):
                # Get bandpass properties
                wavelengths = bandpass.wave  # Bandpass wavelengths in Angstrom
                throughput = bandpass.throughput  # Bandpass throughput

                spec_flux_at_time = sed_surface[i, :]

                # Interpolate the spectrum onto the bandpass wavelength grid
                spec_flux_interp = interp1d(spec_wavelengths, spec_flux_at_time, bounds_error=False, fill_value=0)(wavelengths)

                # Perform the bandpass integrations
                I1 = simps(y=spec_flux_interp * throughput * wavelengths, x=wavelengths)  # Weighted integral
                I2 = simps(y=throughput / wavelengths, x=wavelengths)  # Normalization integral
                fnu = I1 / I2 / c_AAs / (4 * np.pi * dist_cm**2)  # Flux density in erg/s/cm²/Hz

                # Calculate the apparent magnitude
                with np.errstate(divide='ignore'):
                    mAB = -2.5 * np.log10(fnu) - 48.6  # AB magnitude

                # Append the magnitude for this observation
                source_mag.append(mAB)

            #convert from mag to Flam
            sky_fluxes = [10**(-0.4 * (mag + 48.6)) for mag in sky_brightness_mags]  # Flux per arcsec^2
            source_fluxes = [10**(-0.4 * (mag + 48.6)) for mag in source_mag] # Flux

            sky_counts = []
            source_counts = []

            for i, (sky_flux, source_flux, bandpass) in enumerate(zip(sky_fluxes, source_fluxes, bandpasses)):

                # Get bandpass properties
                wavelengths = bandpass.wave  # Bandpass wavelengths in Angstrom
                throughput = bandpass.throughput  # Bandpass throughput

                # Compute sky counts
                bpass_integral = np.trapz(throughput / (wavelengths * 1e-8), wavelengths * 1e-8)  # Throughput normalization
                sky_count = (exp_time * LSST_area / gain / h * sky_flux * bpass_integral * (pix_scale**2))  # Counts for sky per pixel
                sky_counts.append(sky_count)

                # Compute source counts
                source_count = (exp_time * LSST_area / gain / h * source_flux * bpass_integral) # Total counts
                source_counts.append(source_count)


        # Convert counts to numpy arrays
            sky_counts = np.array(sky_counts)
            source_counts = np.array(source_counts)

            # Compute Signal-to-Noise Ratio (SNR)
            snr = source_counts / np.sqrt(source_counts / gain + (sky_counts / gain + readout_noise**2) * n_eff)
            # Compute magnitude errors and add noise
            mag_err = 1.09 / snr
            noisy_mags = source_mag + np.random.normal(0, mag_err)
            photoband = []
            photoflux = []
            photomask = []
            encod = 0
            for flt in filters:
                flt_idx = np.where(filts == flt)[0]
                len_this_band = len(flt_idx)
                photoband_tmp = np.zeros(len_per_filter)
                photoflux_tmp = np.zeros(len_per_filter)
                phototime_tmp = np.zeros(len_per_filter)
                photomask_tmp = np.zeros(len_this_band)
                photoband_tmp[:len_this_band] = encod
                photoflux_tmp[:len_this_band] = noisy_mags[flt_idx]
                phototime_tmp[:len_this_band] = mjd[flt_idx] - start_mjd
                photomask_tmp[:len_this_band] = 1
                photoband.append(photoband_tmp)
                photoflux.append(photoflux_tmp)
                photomask.append(photomask_tmp)
                encod += 1
            photoband = np.array(photoband).flatten()
            photoflux = np.array(photoflux).flatten()
            phototime = np.array(phototime).flatten()
            photomask = np.array(photomask).flatten()
            return photoband, photoflux, phototime, photomask




