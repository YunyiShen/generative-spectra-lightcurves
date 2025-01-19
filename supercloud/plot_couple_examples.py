import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


job = 0
batch = 4 #2 #4
batch_size = 10
results = np.load(f"./samples/posterior_test_photometrycond_job{job}_batch{batch}_size{batch_size}_Ia_Goldstein_centeringFalse_realisticLSST_phase.npz")

gt = results['gt']
posterior = results['posterior_samples']
posterior = posterior.reshape(batch_size, 50,posterior.shape[1])

posterior_mean = np.mean(posterior, axis = 1)
posterior_upper = np.quantile(posterior, 0.975, axis = 1)
posterior_lower = np.quantile(posterior, 0.025, axis = 1)

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

scale = 1.

plt.rcParams['font.size'] = 20 
fig, axs = plt.subplots(2, 5+1, figsize=(20/scale, 5/scale))
axs = axs.flatten()
i = 0

for k in range(10):
    gti = gt[k][mask[k]==1]
    wavelengthi = wavelength[k][mask[k]==1]
    in_range = np.where((wavelengthi > 3000) * (wavelengthi < 8000) )
    gti = gti[in_range]
    wavelengthi = wavelengthi[in_range]
    axs[i].plot(wavelengthi, 
                gti, color='red', label='ground truth')
    #breakpoint()
    
    axs[i].plot(wavelength[0,:], posterior_mean[k], 
                color='blue', label='VDiT')
    axs[i].fill_between(wavelength[0,:], 
                                 posterior_lower[k], 
                                 posterior_upper[k], 
                        color='blue', alpha=0.3)
    if k == 5:
        axs[i].legend()
    if k <5 :
        axs[i].set_title(f'phase: {phase[k]}')
        axs[i].tick_params(axis="x", which="both", 
                           bottom=False, top=False, labelbottom=False)
    if k != 0 and k != 5:
        #tt = 1+1
        axs[i].tick_params(axis="y", which="both", 
                           left=False, right=False, labelleft=False)
    if k == 7:
        axs[i].set_xlabel('wavelength')
    axs[i].set_ylim(-15, -11)
    axs[i].set_xlim(2800, 8200)
    i += 1
    if i % 6 == 5: # plot light curves
        phototimei = phototime[k][photomask[k] == 1]
        photofluxi = photoflux[k][photomask[k] == 1]
        photobandi = np.repeat(np.array([0,1,2,3,4,5]), 10)[photomask[k] == 1]
        for flt in range(6):
            maskflt = photobandi == flt
            axs[i].scatter(phototimei[maskflt], photofluxi[maskflt], label = f'{fltnames[flt]}')
            axs[i].plot(phototimei[maskflt], photofluxi[maskflt])
        if k == 4:
            axs[i].set_title('light curves')
            axs[i].tick_params(axis="x", which="both", 
                           bottom=False, top=False, labelbottom=False)
        if k == 9:
            axs[i].legend(ncol = 2)
            axs[i].set_xlabel('time')
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
inset_ax.plot(wavelengthi, gti, color='red')
inset_ax.fill_between(wavelength[0,:],
                          posterior_lower[1],
                            posterior_upper[1],
                            color='blue', alpha=0.3)

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
in_range = np.where((wavelengthi > 3000) * (wavelengthi < 8000) )
gti = gti[in_range]
wavelengthi = wavelengthi[in_range]
inset_ax.plot(wavelengthi, gti, color='red')
inset_ax.fill_between(wavelength[0,:],
                          posterior_lower[2],
                            posterior_upper[2],
                            color='blue', alpha=0.3)

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
in_range = np.where((wavelengthi > 3000) * (wavelengthi < 8000) )
gti = gti[in_range]
wavelengthi = wavelengthi[in_range]
inset_ax.plot(wavelengthi, gti, color='red')
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
inset_ax.plot(wavelengthi, gti, color='red')
inset_ax.fill_between(wavelength[0,:],
                          posterior_lower[8],
                            posterior_upper[8],
                            color='blue', alpha=0.3)

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
