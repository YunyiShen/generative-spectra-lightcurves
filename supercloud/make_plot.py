import matplotlib.pyplot as plt
import numpy as np

postfix = "" #"_centering"

results = np.load(f'./metrics/test_losses{postfix}.npz')

salt_test_loss = results['salt_test_loss']
vdm_test_loss = results['vdm_test_loss']
vdm_coverage = results['vdm_coverage']
vdm_width = results['vdm_width']
identities = results['identities']
wavelength = results['wavelength']

plt.rcParams['font.size'] = 14 
fig, axes = plt.subplots(5, 2, figsize=(15, 6), sharex=True)

# Adjust spacing between subplots
fig.subplots_adjust(hspace=0)

phase = [-10,0,10,20,30]

for i in range(5):
    mean_salt = np.nanmean(salt_test_loss[i], axis=0)
    sd_salt = np.nanstd(salt_test_loss[i], axis=0)
    axes[4-i, 0].plot(wavelength, mean_salt)
    axes[4-i, 0].fill_between(wavelength, 
                              mean_salt - sd_salt, 
                              mean_salt + sd_salt, alpha=0.5)
    axes[4-i, 0].set_ylabel(f'   Phase {phase[i]}', labelpad=8, 
                            rotation=90, ha='center')
    axes[4-i, 0].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[4-i, 0].tick_params(labelbottom=False)
    rangee = 4 if postfix == "" else 1.75#1.5*np.nanmax(np.abs(mean_salt))
    axes[4-i, 0].set_ylim(-rangee, rangee)


    mean_vdm = np.mean(vdm_test_loss[i], axis=0)
    sd_vdm = np.std(vdm_test_loss[i], axis=0)
    axes[4-i, 1].plot(wavelength, mean_vdm)
    axes[4-i, 1].fill_between(wavelength, 
                              mean_vdm - sd_vdm, 
                              mean_vdm + sd_vdm, alpha=0.5)
    axes[4-i, 1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[4-i, 1].tick_params(labelbottom=False)
    rangee = 1.2 if postfix == "" else 1.2#1.5*np.nanmax(np.abs(vdm_test_loss.mean(axis = (1,2) )))
    axes[4-i, 1].set_ylim(-rangee, rangee)

    
    
    #mean_coverage = np.mean(vdm_coverage[i], axis=0)

    #mean_width = np.mean(vdm_width[i], axis=0)
    #sd_width = np.std(vdm_width[i], axis=0)
fig.text(0., 0.5, 'residual', va='center', rotation='vertical', fontsize=12)
axes[0,0].set_title('SALT3')
axes[0,1].set_title('VDiT posterior mean')
axes[4,0].tick_params(labelbottom=True)
axes[4,1].tick_params(labelbottom=True)

axes[4,0].set_xlabel('Wavelength')
axes[4,1].set_xlabel('Wavelength')

plt.tight_layout(rect=[0.04, 0.04, 1, 1])
plt.show()
plt.savefig(f'./plots/predicting{postfix}.png', dpi = 300)
plt.close()


##### UQ results #####
fig, axes = plt.subplots(5, 2, figsize=(15, 6), sharex=True)

# Adjust spacing between subplots
fig.subplots_adjust(hspace=0)
phase = [-10,0,10,20,30]
print(vdm_coverage.mean())
for i in range(5):
    mean_coverage = np.nanmean(vdm_coverage[i], axis=0)
    axes[4-i, 0].plot(wavelength, mean_coverage)
    axes[4-i, 0].set_ylabel(f'    Phase {phase[i]}', labelpad=8, 
                            rotation=90, ha='center')
    axes[4-i, 0].axhline(0.9, color='gray', linestyle='--', linewidth=0.8)
    axes[4-i, 0].tick_params(labelbottom=False)
    axes[4-i, 0].set_ylim(0,1)


    mean_vdm_width = np.mean(vdm_width[i], axis=0)
    sd_vdm_width = np.std(vdm_width[i], axis=0)
    axes[4-i, 1].plot(wavelength, mean_vdm_width)
    axes[4-i, 1].fill_between(wavelength, 
                              mean_vdm_width - sd_vdm_width, 
                              mean_vdm_width + sd_vdm_width, alpha=0.5)
    #axes[4-i, 1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[4-i, 1].tick_params(labelbottom=False)
    axes[4-i, 1].set_ylim(0,1.4)

    
    
    #mean_coverage = np.mean(vdm_coverage[i], axis=0)

    #mean_width = np.mean(vdm_width[i], axis=0)
    #sd_width = np.std(vdm_width[i], axis=0)
axes[0,0].set_title('VDiT CI coverage')
axes[0,1].set_title('VDiT CI width')
axes[4,0].tick_params(labelbottom=True)
axes[4,1].tick_params(labelbottom=True)

axes[4,0].set_xlabel('Wavelength')
axes[4,1].set_xlabel('Wavelength')

plt.tight_layout(rect=[0.04, 0.04, 1, 1])
plt.show()
plt.savefig(f'./plots/UQ{postfix}.png')





