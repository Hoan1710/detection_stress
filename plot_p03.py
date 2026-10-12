import pandas as pd
import matplotlib.pyplot as plt

bvp_path = r'd:\HMI\bio_signal_3\Dataset\P03\E4\BVP.csv'
ppg_path = r'd:\HMI\bio_signal_3\Dataset\P03\GalaxyWatch\PPG.csv'

# Read data
bvp_df = pd.read_csv(bvp_path)
ppg_df = pd.read_csv(ppg_path)

# BVP from E4 is typically 64Hz. Skip first few seconds which might be 0, plot 10 seconds.
# Let's skip first 60 seconds (64 * 60 = 3840 samples)
bvp_start = 64 * 60
bvp_end = bvp_start + 64 * 10 
bvp_snippet = bvp_df['value'].iloc[bvp_start:bvp_end].values

# PPG from GW is typically 25Hz. Skip first 60 seconds (25 * 60 = 1500 samples)
ppg_start = 25 * 60
ppg_end = ppg_start + 25 * 10
ppg_snippet = ppg_df['ppg'].iloc[ppg_start:ppg_end].values

fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# Plot BVP
axs[0].plot(bvp_snippet, color='blue')
axs[0].set_title('E4 BVP Signal (P03) - 10 seconds snapshot')
axs[0].set_xlabel('Sample (64 Hz)')
axs[0].set_ylabel('BVP')
axs[0].grid(True)

# Plot PPG
axs[1].plot(ppg_snippet, color='red')
axs[1].set_title('Galaxy Watch PPG Signal (P03) - 10 seconds snapshot')
axs[1].set_xlabel('Sample (25 Hz)')
axs[1].set_ylabel('PPG')
axs[1].grid(True)

plt.tight_layout()
plt.savefig(r'd:\HMI\bio_signal_3\p03_signals.png')
print("Plot saved to d:\\HMI\\bio_signal_3\\p03_signals.png")
