import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
sys.path.append("../")

job = 0
batch = 4 #2 #4
batch_size = 10
results = np.load(f"./samples/posterior_test_photometrycond_job{job}_batch{batch}_size{batch_size}_Ia_Goldstein_centeringFalse_realisticLSST_phase.npz")
salt3_res = np.load(f"./samples/salt2_samples_job{job}_batch{batch}_size{batch_size}_Ia_Goldstein_centeringFalse_realisticLSST_phase.npz")

gt = results['gt']
posterior = results['posterior_samples']
posterior = posterior.reshape(batch_size, 50,posterior.shape[1])

posterior_mean = np.mean(posterior, axis = 1)
posterior_upper = np.quantile(posterior, 0.975, axis = 1)
posterior_lower = np.quantile(posterior, 0.025, axis = 1)

salt3 = salt3_res['salt_samples']
salt3_mean = np.mean(salt3, axis = 1)
salt3_upper = np.quantile(salt3, 0.975, axis = 1)
salt3_lower = np.quantile(salt3, 0.025, axis = 1)

wavelength = results['wavelength']
mask = results['mask']

# useful for salt2
phase = results['phase']
photoflux = results['photoflux']
phototime = results['phototime']
photomask = results['photomask']
photoband = results['photoband']
identity = results['identity']
fltnames = ['u', 'g', 'r', 'i', 'z', 'y']

scale = .7

plt.rcParams['font.size'] = 30 
fig, axs = plt.subplots(2, 5+1, figsize=(20/scale, 5/scale))
axs = axs.flatten()
i = 0

for k in range(10):
    phototimei = phototime[k][photomask[k] == 1]
    photofluxi = photoflux[k][photomask[k] == 1]
    photobandi = np.repeat(np.array([0,1,2,3,4,5]), 10)[photomask[k] == 1]

    

    phasei = phase[k]
    gti = gt[k][mask[k]==1]



    wavelengthi = wavelength[k][mask[k]==1]
    in_range = np.where((wavelengthi > 3000) * (wavelengthi < 8000) )
    gti = gti[in_range]
    wavelengthi = wavelengthi[in_range]
    axs[i].plot(wavelengthi, 
                gti, color='red', label='ground truth')
    #breakpoint()
    
    axs[i].plot(wavelength[0,:], posterior_mean[k], 
                color='blue', label='DiTSNe')
    axs[i].fill_between(wavelength[0,:], 
                                 posterior_lower[k], 
                                 posterior_upper[k], 
                        color='blue', alpha=0.3)
    axs[i].plot(wavelength[0,:], salt3_mean[k], 
                color='green', label='SALT3')
    axs[i].fill_between(wavelength[0,:],
                                    salt3_lower[k], 
                                    salt3_upper[k], 
                            color='green', alpha=0.3)

    if k == 5:
        axs[i].legend(fontsize=20)
    if k <5 :
        axs[i].set_title(f'Days after peak: {int(phase[k])}')
        axs[i].tick_params(axis="x", which="both", 
                           bottom=False, top=False, labelbottom=False)
    if k != 0 and k != 5:
        #tt = 1+1
        axs[i].tick_params(axis="y", which="both", 
                           left=False, right=False, labelleft=False)
    else:
        axs[i].set_ylabel('log10 flux')
    if k == 7:
        axs[i].set_xlabel('wavelength (Å)')
    axs[i].set_ylim(-15, -10.5)
    axs[i].set_xlim(2800, 8200)
    i += 1
    if i % 6 == 5: # plot light curves
        for flt in range(6):
            maskflt = photobandi == flt
            axs[i].scatter(phototimei[maskflt], photofluxi[maskflt], label = f'{fltnames[flt]}')
            axs[i].plot(phototimei[maskflt], photofluxi[maskflt])
            axs[i].set_ylabel('Absolute magnitude')
        if k == 4:
            axs[i].set_title('light curves')
            axs[i].tick_params(axis="x", which="both", 
                           bottom=False, top=False, labelbottom=False)
        if k == 9:
            axs[i].legend(ncol = 2)
            axs[i].set_xlabel('time (Days after peak)')
        axs[i].set_ylim((-20.5,-12.5))
        axs[i].invert_yaxis()
        i += 1

# some fine scale features

rect = patches.Rectangle((5500, -13.1), 6500 - 5500, -11.6 - (-13.1),
                         linewidth=2, edgecolor='black', 
                         facecolor='none', linestyle='--')
axs[1].add_patch(rect)
inset_ax = inset_axes(axs[1], width="40%", height="41%", loc="lower center")  # Adjust location and size
inset_ax.plot(wavelength[0,:], posterior_mean[1], 
                color='blue')
gti = gt[1][mask[1]==1]
wavelengthi = wavelength[1][mask[1]==1]
in_range = np.where((wavelengthi > 3000) * (wavelengthi < 8000) )
gti = gti[in_range]
wavelengthi = wavelengthi[in_range]

phototimei = phototime[1][photomask[1] == 1]
photofluxi = photoflux[1][photomask[1] == 1]
photobandi = np.repeat(np.array([0,1,2,3,4,5]), 10)[photomask[1] == 1]
phasei = phase[1]



inset_ax.plot(wavelengthi, gti, color='red')
inset_ax.fill_between(wavelength[0,:],
                          posterior_lower[1],
                            posterior_upper[1],
                            color='blue', alpha=0.3)
inset_ax.plot(wavelength[0,:], salt3_mean[1], 
              color='green')
inset_ax.fill_between(wavelength[0,:],
                            salt3_lower[1], 
                            salt3_upper[1],
                            color='green', alpha=0.3)


inset_ax.set_xlim(5500, 6500)
inset_ax.set_ylim(-13.1, -11.6)

inset_ax.tick_params(axis="x", which="both",
                        bottom=False, top=False, labelbottom=False)
inset_ax.tick_params(axis="y", which="both",
                        left=False, right=False, labelleft=False)


#### 
rect = patches.Rectangle((5500, -13.1), 6500 - 5500, -11.6 - (-13.1),
                         linewidth=2, edgecolor='black', 
                         facecolor='none', linestyle='--')

inset_ax = inset_axes(axs[2], width="40%", height="41%", loc="lower center")  # Adjust location and size
inset_ax.plot(wavelength[0,:], posterior_mean[2], 
                color='blue')
gti = gt[2][mask[2]==1]
wavelengthi = wavelength[2][mask[2]==1]

phototimei = phototime[2][photomask[2] == 1]
photofluxi = photoflux[2][photomask[2] == 1]
photobandi = np.repeat(np.array([0,1,2,3,4,5]), 10)[photomask[2] == 1]
phasei = phase[2]



in_range = np.where((wavelengthi > 3000) * (wavelengthi < 8000) )
gti = gti[in_range]
wavelengthi = wavelengthi[in_range]
inset_ax.plot(wavelengthi, gti, color='red')
inset_ax.fill_between(wavelength[0,:],
                          posterior_lower[2],
                            posterior_upper[2],
                            color='blue', alpha=0.3)
inset_ax.plot(wavelength[0,:], salt3_mean[2], 
              color='green')
inset_ax.fill_between(wavelength[0,:],
                            salt3_lower[2],
                            salt3_upper[2],
                            color='green', alpha=0.3)

inset_ax.set_xlim(5500, 6500)
inset_ax.set_ylim(-13.1, -11.6)

inset_ax.tick_params(axis="x", which="both",
                        bottom=False, top=False, labelbottom=False)
inset_ax.tick_params(axis="y", which="both",
                        left=False, right=False, labelleft=False)


axs[2].add_patch(rect)


########

rect = patches.Rectangle((5500, -13.1 + 0.5), 6500 - 5500, -11.6 - (-13.1),
                         linewidth=2, edgecolor='black', 
                         facecolor='none', linestyle='--')
axs[7].add_patch(rect)
inset_ax = inset_axes(axs[7], width="40%", 
                      height="41%", loc="lower center")  # Adjust location and size
inset_ax.plot(wavelength[0,:], posterior_mean[7], 
                color='blue')
gti = gt[7][mask[7]==1]
wavelengthi = wavelength[7][mask[7]==1]

phototimei = phototime[7][photomask[7] == 1]
photofluxi = photoflux[7][photomask[7] == 1]
photobandi = np.repeat(np.array([0,1,2,3,4,5]), 10)[photomask[7] == 1]
phasei = phase[7]



in_range = np.where((wavelengthi > 3000) * (wavelengthi < 8000) )
gti = gti[in_range]
wavelengthi = wavelengthi[in_range]
inset_ax.plot(wavelengthi, gti, color='red')
inset_ax.fill_between(wavelength[0,:],
                          posterior_lower[7],
                            posterior_upper[7],
                            color='blue', alpha=0.3)
inset_ax.plot(wavelength[0,:], salt3_mean[7], 
              color='green')
inset_ax.fill_between(wavelength[0,:],
                            salt3_lower[7],
                            salt3_upper[7],
                            color='green', alpha=0.3)

inset_ax.set_xlim(5500, 6500)
inset_ax.set_ylim(-13.1 + 0.5, -11.6 + 0.5)

inset_ax.tick_params(axis="x", which="both",
                        bottom=False, top=False, labelbottom=False)
inset_ax.tick_params(axis="y", which="both",
                        left=False, right=False, labelleft=False)





#############
rect = patches.Rectangle((5500, -13.1 + 0.5), 6500 - 5500, -11.6 - (-13.1),
                         linewidth=2, edgecolor='black', 
                         facecolor='none', linestyle='--')
axs[8].add_patch(rect)
inset_ax = inset_axes(axs[8], width="40%", 
                      height="41%", loc="lower center")  # Adjust location and size
inset_ax.plot(wavelength[0,:], posterior_mean[8], 
                color='blue')
gti = gt[8][mask[8]==1]
wavelengthi = wavelength[8][mask[8]==1]
in_range = np.where((wavelengthi > 3000) * (wavelengthi < 8000) )
gti = gti[in_range]
wavelengthi = wavelengthi[in_range]

phototimei = phototime[8][photomask[8] == 1]
photofluxi = photoflux[8][photomask[8] == 1]
photobandi = np.repeat(np.array([0,1,2,3,4,5]), 10)[photomask[8] == 1]
phasei = phase[8]


inset_ax.plot(wavelengthi, gti, color='red')

inset_ax.plot(wavelength[0,:], posterior_mean[8], 
                color='blue')
inset_ax.fill_between(wavelength[0,:],
                          posterior_lower[8],
                            posterior_upper[8],
                            color='blue', alpha=0.3)

inset_ax.plot(wavelength[0,:], salt3_mean[8], 
              color='green')
inset_ax.fill_between(wavelength[0,:],
                            salt3_lower[8],
                            salt3_upper[8],
                            color='green', alpha=0.3)

inset_ax.set_xlim(5500, 6500)
inset_ax.set_ylim(-13.1 + 0.5, -11.6 + 0.5)

inset_ax.tick_params(axis="x", which="both",
                        bottom=False, top=False, labelbottom=False)
inset_ax.tick_params(axis="y", which="both",
                        left=False, right=False, labelleft=False)


plt.show()
plt.tight_layout(rect=[0.05, 0.05, 1.6, 1.6])
#plt.tight_layout()
plt.savefig("./plots/plot_couple_examples.png", bbox_inches="tight", dpi = 300)
plt.close()

################# a zoom in ############

scale = .7

plt.rcParams['font.size'] = 40 
fig, axs = plt.subplots(1, 2, figsize=(20/scale, 5/scale))

## first large one 
gti = gt[2][mask[2]==1]
wavelengthi = wavelength[2][mask[2]==1]

phototimei = phototime[2][photomask[2] == 1]
photofluxi = photoflux[2][photomask[2] == 1]
photobandi = np.repeat(np.array([0,1,2,3,4,5]), 10)[photomask[2] == 1]
phasei = phase[2]


in_range = np.where((wavelengthi > 3000) * (wavelengthi < 8000) )
gti = gti[in_range]
wavelengthi = wavelengthi[in_range]

axs[0].plot(wavelengthi, gti, color='red', label='ground truth', linewidth=5)
axs[0].plot(wavelength[0,:], posterior_mean[2], 
                color='blue', label='DiTSNe', linewidth=5)
axs[0].fill_between(wavelength[0,:], 
                                 posterior_lower[2], 
                                 posterior_upper[2], 
                        color='blue', alpha=0.3)
axs[0].plot(wavelength[0,:], salt3_mean[2], 
              color='green', label='SALT3', linewidth=5)
axs[0].fill_between(wavelength[0,:],
                            salt3_lower[2],
                            salt3_upper[2],
                            color='green', alpha=0.3)

axs[0].legend()
axs[0].set_title(f'Days after peak: {int(phase[2])}')


rect = patches.Rectangle((5500, -13.1), 6500 - 5500, -11.6 - (-13.1),
                         linewidth=2, edgecolor='black', 
                         facecolor='none', linestyle='--')

inset_ax = inset_axes(axs[0], width="27%", height="41%", loc="lower center")  # Adjust location and size
inset_ax.plot(wavelength[0,:], posterior_mean[2], 
                color='blue', linewidth=5)

inset_ax.plot(wavelengthi, gti, color='red', linewidth=5)
inset_ax.fill_between(wavelength[0,:],
                          posterior_lower[2],
                            posterior_upper[2],
                            color='blue', alpha=0.3)
inset_ax.plot(wavelength[0,:], salt3_mean[2], 
              color='green', linewidth=5)
inset_ax.fill_between(wavelength[0,:],
                            salt3_lower[2],
                            salt3_upper[2],
                            color='green', alpha=0.3)

inset_ax.set_xlim(5500, 6500)
inset_ax.set_ylim(-13.1, -11.6)

inset_ax.tick_params(axis="x", which="both",
                        bottom=False, top=False, labelbottom=False)
inset_ax.tick_params(axis="y", which="both",
                        left=False, right=False, labelleft=False)


axs[0].add_patch(rect)

## second large one
gti = gt[7][mask[7]==1]
wavelengthi = wavelength[7][mask[7]==1]
in_range = np.where((wavelengthi > 3000) * (wavelengthi < 8000) )
gti = gti[in_range]
wavelengthi = wavelengthi[in_range]

phototimei = phototime[7][photomask[7] == 1]
photofluxi = photoflux[7][photomask[7] == 1]
photobandi = np.repeat(np.array([0,1,2,3,4,5]), 10)[photomask[7] == 1]
phasei = phase[7]



axs[1].plot(wavelengthi, gti, color='red', label='ground truth', linewidth=5)
axs[1].plot(wavelength[0,:], posterior_mean[7], 
                color='blue', label='DiTSNe', linewidth=5)
axs[1].fill_between(wavelength[0,:], 
                                 posterior_lower[7], 
                                 posterior_upper[7], 
                        color='blue', alpha=0.3)
axs[1].plot(wavelength[0,:], salt3_mean[7], 
              color='green', label='SALT3', linewidth=5)
axs[1].fill_between(wavelength[0,:],
                            salt3_lower[7],
                            salt3_upper[7],
                            color='green', alpha=0.3)#axs[1].legend(fontsize=20)
axs[1].set_title(f'Days after peak: {int(phase[7])}')

axs[1].tick_params(axis="y", which="both",
                        left=False, right=False, labelleft=False)


rect = patches.Rectangle((5500, -13.1 + 0.5), 6500 - 5500, -11.6 - (-13.1),
                         linewidth=2, edgecolor='black', 
                         facecolor='none', linestyle='--')
axs[1].add_patch(rect)
inset_ax = inset_axes(axs[1], width="27%", 
                      height="41%", loc="lower center")  # Adjust location and size
inset_ax.plot(wavelength[0,:], posterior_mean[7], 
                color='blue', linewidth=5)


inset_ax.plot(wavelengthi, gti, color='red', linewidth=5)

inset_ax.plot(wavelength[0,:], salt3_mean[7], 
              color='green', linewidth=5)
inset_ax.fill_between(wavelength[0,:],
                            salt3_lower[7],
                            salt3_upper[7],
                            color='green', alpha=0.3)
inset_ax.plot(wavelength[0,:], posterior_mean[7], 
                color='blue', linewidth=5)
inset_ax.fill_between(wavelength[0,:],
                          posterior_lower[7],
                            posterior_upper[7],
                            color='blue', alpha=0.3)

inset_ax.set_xlim(5500, 6500)
inset_ax.set_ylim(-13.1 + 0.5, -11.6 + 0.5)

inset_ax.tick_params(axis="x", which="both",
                        bottom=False, top=False, labelbottom=False)
inset_ax.tick_params(axis="y", which="both",
                        left=False, right=False, labelleft=False)

for i in range(2):
    axs[i].set_ylim(-15, -10.5)
    axs[i].set_xlim(2800, 8200)
    axs[i].set_xlabel('wavelength (Å)')
axs[0].set_ylabel('log10 flux')

plt.show()
plt.tight_layout(rect=[0.05, 0.05, 1.6, 1.6])
#plt.tight_layout()
plt.savefig("./plots/plot_couple_examples_zoomin.png", bbox_inches="tight", dpi = 300)
