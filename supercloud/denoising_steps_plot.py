import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

results = np.load("example_denoising_steps.npz")
gt = results['gt']
posterior = results['sample']
wavelength = results['wavelength']

fig, ax = fig, axes = plt.subplots(5, 10, figsize=(50, 20))

for i in range(5): # i for phase
    for j in range(10): # steps 
        ax[i, j].plot(wavelength[i], posterior[j * 20, i], color='blue', linewidth=2)
        ax[i, j].plot(wavelength[i], gt[i], color='red', linewidth=2)
        ax[i, j].set_ylabel('Absolute magnitude')
    if i == 4:
        ax[i, j].set_xlabel('Wavelength (Å)')

plt.show()
# Save the figure
fig.savefig('./plots/denoising_steps_plot.jpg', dpi=300, bbox_inches='tight')


# make a video
which = 1
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot(wavelength[0], posterior[0,which], color='blue', label="DiTSNE", linewidth=2)
gt_line, = ax.plot(wavelength[0], gt[which], color='red', label="ground truth", linewidth=2)
ax.set_xlabel('Wavelength (Å)')
ax.set_ylabel('Absolute magnitude')
ax.legend(loc='lower left')
frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


def update(frame):
    y = posterior[frame, which]
    line.set_ydata(y)
    combined_y = np.concatenate([y, gt[1]])
    # Adjust y-limits dynamically
    margin = 0.1 * (np.max(combined_y) - np.min(combined_y) + 1e-5)
    #ax.set_ylim(np.min(combined_y) - margin, np.max(combined_y) + margin)
    ax.set_ylim(-14, -11)
    frame_text.set_text(f"denoising time: {1.-frame/posterior.shape[0]:.2f}")
    
    return line, gt_line, frame_text

ani = FuncAnimation(fig, update, frames=posterior.shape[0], blit=False)
ani.save("./plots/denoising.mp4", writer="ffmpeg", fps=30)


#breakpoint()


