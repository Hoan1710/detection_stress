import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = r'd:\HMI\bio_signal_3\IMAT\P03_processed_GD_denoised.csv'

# Read just the columns we need to save memory
df = pd.read_csv(csv_path, usecols=['session', 'denoisedGalaxy', 'denoisedE4'])

unique_sessions = df['session'].unique()
print(f"Available sessions: {unique_sessions}")

# Identify non-stress and stress sessions
non_stress_sess = [s for s in unique_sessions if 'baseline' in s.lower() or 'adaptation' in s.lower()]
stress_sess = [s for s in unique_sessions if 'stress' in s.lower() or 'tsst' in s.lower() or 'math' in s.lower()]

if not non_stress_sess:
    non_stress_sess = [unique_sessions[0]] # fallback to first
if not stress_sess:
    stress_sess = [unique_sessions[-1]] # fallback to last

ns_session = non_stress_sess[0]
s_session = stress_sess[0]

# Get first window for each
ns_window = df[df['session'] == ns_session].iloc[0]
s_window = df[df['session'] == s_session].iloc[0]

def parse_signal(sig_str):
    if pd.isna(sig_str):
        return []
    return np.array([float(x) for x in sig_str.split(';')])

ns_galaxy = parse_signal(ns_window['denoisedGalaxy'])
ns_e4 = parse_signal(ns_window['denoisedE4'])

s_galaxy = parse_signal(s_window['denoisedGalaxy'])
s_e4 = parse_signal(s_window['denoisedE4'])

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'P03 IMAT Denoised Signals\nNon-Stress ({ns_session}) vs Stress ({s_session})', fontsize=16)

# Non-Stress Galaxy
axs[0, 0].plot(ns_galaxy, color='blue')
axs[0, 0].set_title(f'Galaxy PPG - Non-Stress')
axs[0, 0].grid(True)

# Non-Stress E4
axs[1, 0].plot(ns_e4, color='lightblue')
axs[1, 0].set_title(f'E4 BVP - Non-Stress')
axs[1, 0].grid(True)

# Stress Galaxy
axs[0, 1].plot(s_galaxy, color='red')
axs[0, 1].set_title(f'Galaxy PPG - Stress')
axs[0, 1].grid(True)

# Stress E4
axs[1, 1].plot(s_e4, color='salmon')
axs[1, 1].set_title(f'E4 BVP - Stress')
axs[1, 1].grid(True)

plt.tight_layout()
out_path = r'd:\HMI\bio_signal_3\p03_imat_signals.png'
plt.savefig(out_path)
print(f"Plot saved to {out_path}")
