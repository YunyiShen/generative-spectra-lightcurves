import numpy as np
import matplotlib.pyplot as plt
import re

def get_goldstein_params(filename):
    params = re.findall(r'[-+]?\d*\.\d+e[-+]?\d+', filename)
    params = [float(i) for i in params]
    return np.array(params)


postfix =""

results = np.load(f'./metrics/test_losses{postfix}.npz')
coverage_at_phase = np.load(f'./metrics/coverage_varing.npz')  
coverage_varing = coverage_at_phase['coverage'].mean(axis = (2,3))
coverage_varing_salt = coverage_at_phase['coverage_salt3'].mean(axis = (2,3))
coverage_varing_salt_recon = coverage_at_phase['coverage_salt3_recon'].mean(axis = (2,3))

levels = 1.- coverage_at_phase['alphas']

salt_test_loss = results['salt_test_loss'].mean(axis = 2)
salt_coverage = results['salt_coverage'].mean(axis = 2)
salt_width = results['salt_width'].mean(axis = 2)


#breakpoint()

vdm_test_loss = results['vdm_test_loss'].mean(axis = 2)
vdm_coverage = results['vdm_coverage'].mean(axis = 2)
vdm_width = results['vdm_width'].mean(axis = 2)

salt_recon_test_loss = results['salt_recon_test_loss'].mean(axis = 2)
salt_recon_coverage = results['salt_recon_coverage'].mean(axis = 2)
salt_recon_width = results['salt_recon_width'].mean(axis = 2)

identities = results['identities']


param = np.array([[get_goldstein_params(identities[i,j]) for i in range(identities.shape[0])] for j in range(identities.shape[1])])
param = np.swapaxes(param, 0, 1)


plt.rcParams['font.size'] = 30
fig, axes = plt.subplots(6, 5, figsize=(35, 30), 
                         sharex=False)

# Adjust spacing between subplots
fig.subplots_adjust(hspace=0)

phase = [-10,0,10,20,30]
param_names = ['Energy', "Mass (mSum)", "mNi+Fe (mSum)", "mIME (mSum)", "mCO (mSum)", ""]


n_bins = 15
for i in range(5):# 5 phases
    for j in range(6): # 6 parameters
        param_this_setup = param[i,:,j]
        vdm_test_loss_this = vdm_test_loss[i]
        salt_test_loss_this = salt_test_loss[i]
        salt_recon_test_loss_this = salt_recon_test_loss[i]
        bin_edges = np.quantile(param_this_setup, np.linspace(0, 1, n_bins + 1)) #np.linspace(param_this_setup.min(), 0.8 * param_this_setup.max(), num=20)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_indices = np.digitize(param_this_setup, bin_edges) - 1

        mean_salt = np.array([np.nanmean(salt_test_loss_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])
        sd_salt  = np.array([np.nanstd(salt_test_loss_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])

        mean_salt_recon = np.array([np.nanmean(salt_recon_test_loss_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])
        sd_salt_recon  = np.array([np.nanstd(salt_recon_test_loss_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])
        
        mean_vdm = np.array([np.nanmean(vdm_test_loss_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])
        sd_vdm  = np.array([np.nanstd(vdm_test_loss_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])

        #breakpoint()

        if i == 0 and j == 0:
            axes[j, i].plot(bin_centers, mean_salt,color='green', label='SALT3', linewidth=2)
        else:
            axes[j, i].plot(bin_centers, mean_salt,color='green', linewidth=2)
        axes[j, i].fill_between(bin_centers, 
                              mean_salt - sd_salt, 
                              mean_salt + sd_salt,color='green', alpha=0.3)
    
        axes[0, i].set_title(f'Days after peak: {phase[i]}')
        axes[j, i].axhline(0, color='red', linestyle='--', linewidth=1.5)
        #axes[j, i].tick_params(labelbottom=False)
        rangee = 2.2 if postfix == "" else 1.75#1.5*np.nanmax(np.abs(mean_salt))
        axes[j, i].set_ylim(-rangee, rangee)



        if i == 0 and j==0:
            axes[j, i].plot(bin_centers, mean_vdm,color='blue', label='DiTSNe', linewidth=2)
        else:
            axes[j, i].plot(bin_centers, mean_vdm,color='blue', linewidth=2)
        axes[j, i].fill_between(bin_centers, 
                              mean_vdm - sd_vdm, 
                              mean_vdm + sd_vdm, 
                              color='blue', alpha=0.3)
    
        
        if i == 0 and j == 0:
            axes[j, i].plot(bin_centers, mean_salt_recon,color='purple', label='SALT3-spectra', linewidth=2)
        else:
            axes[j, i].plot(bin_centers, mean_salt_recon,color='purple', linewidth=2)
        axes[j, i].fill_between(bin_centers,
                              mean_salt_recon - sd_salt_recon,
                              mean_salt_recon + sd_salt_recon, color='purple', alpha=0.3)
        
        axes[j, i].set_xlabel(param_names[j])
        

        axes[j,0].set_ylabel('residual')\

#fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.2, 0.785), handlelength=1) 
axes[0,0].legend(loc = "lower center")
plt.tight_layout(rect=[0.0, 0.0, 1., 1.])
plt.show()
plt.savefig("./plots/physical_param_residual.png", dpi = 300)
plt.close()


##########################################
############### coverage #################
##########################################
fig, axes = plt.subplots(6, 5, figsize=(35, 30), 
                         sharex=False)

# Adjust spacing between subplots
fig.subplots_adjust(hspace=0)
for i in range(5):# 5 phases
    for j in range(6): # 6 parameters
        param_this_setup = param[i,:,j]
        vdm_coverage_this = vdm_coverage[i]
        salt_coverage_this = salt_coverage[i]
        salt_recon_coverage_this = salt_recon_coverage[i]
        bin_edges = np.quantile(param_this_setup, np.linspace(0, 1, n_bins + 1)) #np.linspace(param_this_setup.min(), 0.8 * param_this_setup.max(), num=20)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_indices = np.digitize(param_this_setup, bin_edges) - 1

        mean_salt = np.array([np.nanmean(salt_coverage_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])
        sd_salt  = np.array([np.nanstd(salt_coverage_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])

        mean_salt_recon = np.array([np.nanmean(salt_recon_coverage_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])
        sd_salt_recon  = np.array([np.nanstd(salt_recon_coverage_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])
        
        mean_vdm = np.array([np.nanmean(vdm_coverage_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])
        sd_vdm  = np.array([np.nanstd(vdm_coverage_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])

        #breakpoint()

        if i == 0 and j == 0:
            axes[j, i].plot(bin_centers, mean_salt,color='green', label='SALT3', linewidth=2)
        else:
            axes[j, i].plot(bin_centers, mean_salt,color='green', linewidth=2)
        axes[j, i].fill_between(bin_centers, 
                              mean_salt - sd_salt, 
                              mean_salt + sd_salt,color='green', alpha=0.3)
    
        axes[0, i].set_title(f'Days after peak: {phase[i]}')
        axes[j, i].axhline(0, color='red', linestyle='--', linewidth=1.5)
        #axes[j, i].tick_params(labelbottom=False)
        
        axes[j, i].set_ylim(0.01,1.05)



        if i == 0 and j==0:
            axes[j, i].plot(bin_centers, mean_vdm,color='blue', label='DiTSNe', linewidth=2)
        else:
            axes[j, i].plot(bin_centers, mean_vdm,color='blue', linewidth=2)
        axes[j, i].fill_between(bin_centers, 
                              mean_vdm - sd_vdm, 
                              mean_vdm + sd_vdm, 
                              color='blue', alpha=0.3)
    
        
        if i == 0 and j == 0:
            axes[j, i].plot(bin_centers, mean_salt_recon,color='purple', label='SALT3-spectra', linewidth=2)
        else:
            axes[j, i].plot(bin_centers, mean_salt_recon,color='purple', linewidth=2)
        axes[j, i].fill_between(bin_centers,
                              mean_salt_recon - sd_salt_recon,
                              mean_salt_recon + sd_salt_recon, color='purple', alpha=0.3)
        
        axes[j, i].set_xlabel(param_names[j])
        

        axes[j,0].set_ylabel('coverage')\

#fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.2, 0.785), handlelength=1) 
axes[0,0].legend(loc = "lower center")
plt.tight_layout(rect=[0.0, 0.0, 1., 1.])
plt.show()
plt.savefig("./plots/physical_param_coverage.png", dpi = 300)
plt.close()

##########################################
############### width #################
##########################################
fig, axes = plt.subplots(6, 5, figsize=(35, 30), 
                         sharex=False)

# Adjust spacing between subplots
fig.subplots_adjust(hspace=0)
for i in range(5):# 5 phases
    for j in range(6): # 6 parameters
        param_this_setup = param[i,:,j]
        vdm_width_this = vdm_width[i]
        salt_width_this = salt_width[i]
        salt_recon_width_this = salt_recon_width[i]
        bin_edges = np.quantile(param_this_setup, np.linspace(0, 1, n_bins + 1)) #np.linspace(param_this_setup.min(), 0.8 * param_this_setup.max(), num=20)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_indices = np.digitize(param_this_setup, bin_edges) - 1

        mean_salt = np.array([np.nanmean(salt_width_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])
        sd_salt  = np.array([np.nanstd(salt_width_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])

        mean_salt_recon = np.array([np.nanmean(salt_recon_width_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])
        sd_salt_recon  = np.array([np.nanstd(salt_recon_width_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])
        
        mean_vdm = np.array([np.nanmean(vdm_width_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])
        sd_vdm  = np.array([np.nanstd(vdm_width_this[bin_indices == k]) if np.any(bin_indices == k) else np.nan for k in range(len(bin_centers))])

        #breakpoint()

        if i == 0 and j == 0:
            axes[j, i].plot(bin_centers, mean_salt,color='green', label='SALT3', linewidth=2)
        else:
            axes[j, i].plot(bin_centers, mean_salt,color='green', linewidth=2)
        axes[j, i].fill_between(bin_centers, 
                              mean_salt - sd_salt, 
                              mean_salt + sd_salt,color='green', alpha=0.3)
    
        axes[0, i].set_title(f'Days after peak: {phase[i]}')
        axes[j, i].axhline(0, color='red', linestyle='--', linewidth=1.5)
        #axes[j, i].tick_params(labelbottom=False)
        axes[j, i].set_ylim(0,1.4)



        if i == 0 and j==0:
            axes[j, i].plot(bin_centers, mean_vdm,color='blue', label='DiTSNe', linewidth=2)
        else:
            axes[j, i].plot(bin_centers, mean_vdm,color='blue', linewidth=2)
        axes[j, i].fill_between(bin_centers, 
                              mean_vdm - sd_vdm, 
                              mean_vdm + sd_vdm, 
                              color='blue', alpha=0.3)
    
        
        if i == 0 and j == 0:
            axes[j, i].plot(bin_centers, mean_salt_recon,color='purple', label='SALT3-spectra', linewidth=2)
        else:
            axes[j, i].plot(bin_centers, mean_salt_recon,color='purple', linewidth=2)
        axes[j, i].fill_between(bin_centers,
                              mean_salt_recon - sd_salt_recon,
                              mean_salt_recon + sd_salt_recon, color='purple', alpha=0.3)
        
        axes[j, i].set_xlabel(param_names[j])
        

        axes[j,0].set_ylabel('CI width')\

#fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.2, 0.785), handlelength=1) 
axes[0,0].legend(loc = "lower center")
plt.tight_layout(rect=[0.0, 0.0, 1., 1.])
plt.show()
plt.savefig("./plots/physical_param_width.png", dpi = 300)
plt.close()

