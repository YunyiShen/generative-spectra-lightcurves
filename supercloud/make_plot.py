import matplotlib.pyplot as plt
import numpy as np

postfix =""

results = np.load(f'./metrics/test_losses{postfix}.npz')
coverage_at_phase = np.load(f'./metrics/coverage_varing.npz')  
coverage_varing = coverage_at_phase['coverage'].mean(axis = (2,3))
coverage_varing_salt = coverage_at_phase['coverage_salt3'].mean(axis = (2,3))
coverage_varing_salt_recon = coverage_at_phase['coverage_salt3_recon'].mean(axis = (2,3))

levels = 1.- coverage_at_phase['alphas']

salt_test_loss = results['salt_test_loss']
salt_coverage = results['salt_coverage']
salt_width = results['salt_width']


#breakpoint()

vdm_test_loss = results['vdm_test_loss']
vdm_coverage = results['vdm_coverage']
vdm_width = results['vdm_width']

salt_recon_test_loss = results['salt_recon_test_loss']
salt_recon_coverage = results['salt_recon_coverage']
salt_recon_width = results['salt_recon_width']

identities = results['identities']
wavelength = results['wavelength']

all_mean_test_salt = []
all_mean_coverage_salt = []
all_mean_width_salt = []
all_mean_test_vdm = []
all_mean_coverage = []
all_mean_width = []

all_mean_test_salt_recon = []
all_mean_coverage_salt_recon = []
all_mean_width_salt_recon = []



plt.rcParams['font.size'] = 30
fig, axes = plt.subplots(4, 5, figsize=(30, 18), 
                         sharex=False)

# Adjust spacing between subplots
fig.subplots_adjust(hspace=0)

phase = [-10,0,10,20,30]

for i in range(5):
    mean_salt = np.nanmean(salt_test_loss[i], axis=0)
    sd_salt = np.nanstd(salt_test_loss[i], axis=0)
    if i == 0:
        axes[0, i].plot(wavelength, mean_salt,color='green', label='SALT3', linewidth=2)
    else:
        axes[0, i].plot(wavelength, mean_salt,color='green', linewidth=2)
    axes[0, i].fill_between(wavelength, 
                              mean_salt - sd_salt, 
                              mean_salt + sd_salt,color='green', alpha=0.3)
    #axes[0, 4-i].set_ylabel(f'   Phase {phase[i]}', labelpad=8, 
    #                        rotation=90, ha='center')
    axes[0, i].set_title(f'Days after peak: {phase[i]}')
    axes[0, i].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0, i].tick_params(labelbottom=False)
    rangee = 2.2 if postfix == "" else 1.75#1.5*np.nanmax(np.abs(mean_salt))
    axes[0, i].set_ylim(-rangee, rangee)


    mean_vdm = np.mean(vdm_test_loss[i], axis=0)
    sd_vdm = np.std(vdm_test_loss[i], axis=0)

    if i == 0:
        axes[0, i].plot(wavelength, mean_vdm,color='blue', label='DiTSNe', linewidth=2)
    else:
        axes[0, i].plot(wavelength, mean_vdm,color='blue', linewidth=2)
    axes[0, i].fill_between(wavelength, 
                              mean_vdm - sd_vdm, 
                              mean_vdm + sd_vdm, 
                              color='blue', alpha=0.3)
    #axes[1, i].axhline(0, color='red', linestyle='--', linewidth=1.5)
    #axes[1, i].tick_params(labelbottom=True)
    #rangee = 1.2 if postfix == "" else 1.2#1.5*np.nanmax(np.abs(vdm_test_loss.mean(axis = (1,2) )))
    #axes[1, i].set_ylim(-rangee, rangee)
    mean_salt_recon = np.nanmean(salt_recon_test_loss[i], axis=0)
    sd_salt_recon = np.nanstd(salt_recon_test_loss[i], axis=0)
    if i == 0:
        axes[0, i].plot(wavelength, mean_salt_recon,color='purple', label='SALT3-spectra', linewidth=2)
    else:
        axes[0, i].plot(wavelength, mean_salt_recon,color='purple', linewidth=2)
    axes[0, i].fill_between(wavelength,
                              mean_salt_recon - sd_salt_recon,
                              mean_salt_recon + sd_salt_recon, color='purple', alpha=0.3)
    
    #mean_coverage = np.mean(vdm_coverage[i], axis=0)

    #mean_width = np.mean(vdm_width[i], axis=0)
    #sd_width = np.std(vdm_width[i], axis=0)
    all_mean_test_salt.append( float("{:.3g}".format( ( np.nanmean(salt_test_loss[i] ** 2)))))
    all_mean_test_vdm.append(float("{:.3g}".format(( np.nanmean(vdm_test_loss[i] ** 2)))))
    all_mean_test_salt_recon.append(float("{:.3g}".format(( np.nanmean(salt_recon_test_loss[i] ** 2)))))

all_mean_test_salt.append( float("{:.3g}".format( ( np.nanmean(salt_test_loss ** 2)))))
all_mean_test_vdm.append(float("{:.3g}".format(( np.nanmean(vdm_test_loss ** 2)))))
all_mean_test_salt_recon.append(float("{:.3g}".format(( np.nanmean(salt_recon_test_loss ** 2)))))

all_mean_test_salt.append( float("{:.3g}".format( ( np.nanmean(salt_test_loss[1:] ** 2)))))
all_mean_test_vdm.append(float("{:.3g}".format(( np.nanmean(vdm_test_loss[1:] ** 2)))))
all_mean_test_salt_recon.append(float("{:.3g}".format(( np.nanmean(salt_recon_test_loss[1:] ** 2)))))

#fig.text(0.02, 0.5, 'residual', va='center', rotation='vertical', fontsize=18)
axes[0,0].set_ylabel('residual')
#axes[0,0].legend()
#axes[1,0].set_ylabel('DiTSNe')
#axes[0,1].set_title('DiTSNe posterior mean')
#axes[4,0].tick_params(labelbottom=True)
#axes[4,1].tick_params(labelbottom=True)
fig.subplots_adjust(bottom=0.01) 
fig.subplots_adjust(left=0.03) 
axes[2,2].set_xlabel('Wavelength (Å)')
#axes[4,1].set_xlabel('Wavelength')

#plt.tight_layout(rect=[0.04, 0.04, 1, 1])
#plt.show()
#plt.savefig(f'./plots/predicting{postfix}.png', dpi = 300)
#plt.close()


##### UQ results #####
#fig, axes = plt.subplots(2, 5, figsize=(18, 6), sharex=False)

# Adjust spacing between subplots
#fig.subplots_adjust(hspace=0)
phase = [-10,0,10,20,30]
#print(vdm_coverage.mean())
for i in range(5):
    mean_coverage = np.nanmean(vdm_coverage[i], axis=0)
    mean_salt_coverage = np.nanmean(salt_coverage[i], axis=0)
    mean_salt_recon_coverage = np.nanmean(salt_recon_coverage[i], axis=0)

    axes[1, i].plot(wavelength, mean_coverage, color='blue')
    axes[1, i].plot(wavelength, mean_salt_coverage, color='green')
    axes[1, i].plot(wavelength, mean_salt_recon_coverage, color='purple')
    #axes[0, i].set_ylabel(f'    Phase {phase[i]}', labelpad=8, 
    #                        rotation=90, ha='center')
    #axes[0, i].set_title(f'Phase {phase[i]}')
    axes[1, i].axhline(0.9, color='red', linestyle='--', linewidth=1.5)
    axes[1, i].tick_params(labelbottom=False)
    axes[1, i].set_ylim(0.01,1.05)


    mean_vdm_width = np.nanmean(vdm_width[i], axis=0)
    sd_vdm_width = np.nanstd(vdm_width[i], axis=0)

    mean_salt_width = np.nanmean(salt_width[i], axis=0)
    sd_salt_width = np.nanstd(salt_width[i], axis=0)

    mean_salt_recon_width = np.nanmean(salt_recon_width[i], axis=0)
    sd_salt_recon_width = np.nanstd(salt_recon_width[i], axis=0)

    axes[2, i].plot(wavelength, mean_vdm_width,
                    color='blue')
    axes[2, i].fill_between(wavelength, 
                              mean_vdm_width - sd_vdm_width, 
                              mean_vdm_width + sd_vdm_width, 
                              color='blue',
                              alpha=0.3)
    axes[2, i].plot(wavelength, mean_salt_width,
                    color='green')
    axes[2, i].fill_between(wavelength, 
                              mean_salt_width - sd_salt_width, 
                              mean_salt_width + sd_salt_width, 
                              color='green',
                              alpha=0.3)
    
    axes[2, i].plot(wavelength, mean_salt_recon_width,
                    color='purple')
    axes[2, i].fill_between(wavelength, 
                              mean_salt_recon_width - sd_salt_recon_width, 
                              mean_salt_recon_width + sd_salt_recon_width, 
                              color='purple',
                              alpha=0.3)
    #axes[1, i].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[2, i].tick_params(labelbottom=True)
    axes[2, i].set_ylim(0,1.4)

    all_mean_width.append(float("{:.3g}".format(np.nanmean(vdm_width[i]))))
    all_mean_coverage.append(float("{:.3g}".format(mean_coverage.mean())))
    all_mean_coverage_salt.append(float("{:.3g}".format(mean_salt_coverage.mean())))
    all_mean_width_salt.append(float("{:.3g}".format(mean_salt_width.mean())))

    all_mean_coverage_salt_recon.append(float("{:.3g}".format(mean_salt_recon_coverage.mean())))
    all_mean_width_salt_recon.append(float("{:.3g}".format(mean_salt_recon_width.mean())))



all_mean_width.append(float("{:.3g}".format(vdm_width.mean())))
all_mean_coverage.append(float("{:.3g}".format(vdm_coverage.mean())))
all_mean_coverage_salt.append(float("{:.3g}".format(salt_coverage.mean())))
all_mean_width_salt.append(float("{:.3g}".format(np.nanmean( salt_width))))
all_mean_coverage_salt_recon.append(float("{:.3g}".format(salt_recon_coverage.mean())))
all_mean_width_salt_recon.append(float("{:.3g}".format(np.nanmean( salt_recon_width))))

all_mean_width.append(float("{:.3g}".format(vdm_width[1:].mean())))
all_mean_coverage.append(float("{:.3g}".format(vdm_coverage[1:].mean())))
all_mean_coverage_salt.append(float("{:.3g}".format(salt_coverage[1:].mean())))
all_mean_width_salt.append(float("{:.3g}".format(np.nanmean( salt_width[1:]))))
all_mean_coverage_salt_recon.append(float("{:.3g}".format(salt_recon_coverage[1:].mean())))
all_mean_width_salt_recon.append(float("{:.3g}".format(np.nanmean( salt_recon_width[1:]))))

    
    
    #mean_coverage = np.mean(vdm_coverage[i], axis=0)

    #mean_width = np.mean(vdm_width[i], axis=0)
    #sd_width = np.std(vdm_width[i], axis=0)
axes[1,0].set_ylabel('CI coverage')
axes[2,0].set_ylabel('CI width')
#axes[4,0].tick_params(labelbottom=True)
#axes[4,1].tick_params(labelbottom=True)

#axes[4,0].set_xlabel('Wavelength')
#axes[1,2].set_xlabel('Wavelength (Å)')
plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce width & height spacing

x = np.linspace(0., 1., 100)

for i in range(5):
    
    axes[3, i].plot(x, x, color='red', linestyle='--')
    axes[3, i].plot(levels, coverage_varing[:,i], color='blue', linewidth=2)
    axes[3, i].plot(levels, coverage_varing_salt[:,i], color='green', linewidth=2)
    axes[3, i].plot(levels, coverage_varing_salt_recon[:,i], color='purple', linewidth=2)
    #axes[3, i].tick_params(labelbottom=False)
    axes[3, i].set_ylim(0.0, .95)
    axes[3, i].set_xlim(0.05, .95)

axes[3, 0].set_ylabel('CI coverage')
axes[3, 2].tick_params(labelbottom=True)
axes[3, 2].set_xlabel('nominal coverage')
fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.2, 0.785), handlelength=1) 


plt.tight_layout(rect=[0.0, 0.0, 1., 1.])
plt.show()
plt.savefig(f'./plots/both{postfix}.png')


print("salt_test: ",all_mean_test_salt)
print("vdm_test: ",all_mean_test_vdm)
print("coverage: ",all_mean_coverage)
print("salt_coverage: ",all_mean_coverage_salt)
print("width: ",all_mean_width)
print("salt_width: ",all_mean_width_salt)
print("salt_recon: ",all_mean_test_salt_recon)
print("salt_recon_coverage: ",all_mean_coverage_salt_recon)
print("salt_recon_width: ",all_mean_width_salt_recon)


