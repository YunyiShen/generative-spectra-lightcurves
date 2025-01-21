import matplotlib.pyplot as plt
import numpy as np

postfix = "_centering"

results = np.load(f'./metrics/test_losses{postfix}.npz')

salt_test_loss = results['salt_test_loss']
vdm_test_loss = results['vdm_test_loss']
vdm_coverage = results['vdm_coverage']
vdm_width = results['vdm_width']
identities = results['identities']
wavelength = results['wavelength']

plt.rcParams['font.size'] = 14 
fig, axes = plt.subplots(2, 5, figsize=(18, 6), sharex=False)

# Adjust spacing between subplots
fig.subplots_adjust(hspace=0)

phase = [-10,0,10,20,30]

for i in range(5):
    mean_salt = np.nanmean(salt_test_loss[i], axis=0)
    sd_salt = np.nanstd(salt_test_loss[i], axis=0)
    axes[0, i].plot(wavelength, mean_salt,color='blue')
    axes[0, i].fill_between(wavelength, 
                              mean_salt - sd_salt, 
                              mean_salt + sd_salt,color='blue', alpha=0.3)
    #axes[0, 4-i].set_ylabel(f'   Phase {phase[i]}', labelpad=8, 
    #                        rotation=90, ha='center')
    axes[0, i].set_title(f'Phase {phase[i]}')
    axes[0, i].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0, i].tick_params(labelbottom=False)
    rangee = 4 if postfix == "" else 1.75#1.5*np.nanmax(np.abs(mean_salt))
    axes[0, i].set_ylim(-rangee, rangee)


    mean_vdm = np.mean(vdm_test_loss[i], axis=0)
    sd_vdm = np.std(vdm_test_loss[i], axis=0)
    axes[1, i].plot(wavelength, mean_vdm,color='blue')
    axes[1, i].fill_between(wavelength, 
                              mean_vdm - sd_vdm, 
                              mean_vdm + sd_vdm, 
                              color='blue', alpha=0.3)
    axes[1, i].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1, i].tick_params(labelbottom=True)
    rangee = 1.2 if postfix == "" else 1.2#1.5*np.nanmax(np.abs(vdm_test_loss.mean(axis = (1,2) )))
    axes[1, i].set_ylim(-rangee, rangee)

    
    
    #mean_coverage = np.mean(vdm_coverage[i], axis=0)

    #mean_width = np.mean(vdm_width[i], axis=0)
    #sd_width = np.std(vdm_width[i], axis=0)
fig.text(0.02, 0.5, 'residual', va='center', rotation='vertical', fontsize=18)
axes[0,0].set_ylabel('SALT3')
axes[1,0].set_ylabel('VDiT')
#axes[0,1].set_title('VDiT posterior mean')
#axes[4,0].tick_params(labelbottom=True)
#axes[4,1].tick_params(labelbottom=True)

axes[1,2].set_xlabel('Wavelength (Å)', fontsize=18)
#axes[4,1].set_xlabel('Wavelength')

plt.tight_layout(rect=[0.04, 0.04, 1, 1])
plt.show()
plt.savefig(f'./plots/predicting{postfix}.png', dpi = 300)
plt.close()


##### UQ results #####
fig, axes = plt.subplots(2, 5, figsize=(18, 6), sharex=False)

# Adjust spacing between subplots
fig.subplots_adjust(hspace=0)
phase = [-10,0,10,20,30]
print(vdm_coverage.mean())
for i in range(5):
    mean_coverage = np.nanmean(vdm_coverage[i], axis=0)
    axes[0, i].plot(wavelength, mean_coverage, color='blue')
    #axes[0, i].set_ylabel(f'    Phase {phase[i]}', labelpad=8, 
    #                        rotation=90, ha='center')
    axes[0, i].set_title(f'Phase {phase[i]}')
    axes[0, i].axhline(0.9, color='red', linestyle='--', linewidth=1.5)
    axes[0, i].tick_params(labelbottom=False)
    axes[0, i].set_ylim(0.3,1.05)


    mean_vdm_width = np.mean(vdm_width[i], axis=0)
    sd_vdm_width = np.std(vdm_width[i], axis=0)
    axes[1, i].plot(wavelength, mean_vdm_width,
                    color='blue')
    axes[1, i].fill_between(wavelength, 
                              mean_vdm_width - sd_vdm_width, 
                              mean_vdm_width + sd_vdm_width, 
                              color='blue',
                              alpha=0.3)
    #axes[1, i].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1, i].tick_params(labelbottom=True)
    axes[1, i].set_ylim(0,1.4)

    
    
    #mean_coverage = np.mean(vdm_coverage[i], axis=0)

    #mean_width = np.mean(vdm_width[i], axis=0)
    #sd_width = np.std(vdm_width[i], axis=0)
axes[0,0].set_ylabel('VDiT CI coverage')
axes[1,0].set_ylabel('VDiT CI width')
#axes[4,0].tick_params(labelbottom=True)
#axes[4,1].tick_params(labelbottom=True)

#axes[4,0].set_xlabel('Wavelength')
axes[1,2].set_xlabel('Wavelength (Å)')

plt.tight_layout(rect=[0.04, 0.04, 1, 1])
plt.show()
plt.savefig(f'./plots/UQ{postfix}.png')





