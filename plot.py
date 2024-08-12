import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
file_path = 'model01_view200'
data = pd.read_csv(file_path + ".csv", header=None, delimiter=' ')

# Randomly sample 10 rows
data = data.drop(columns=[0,1])
data = data.apply(lambda x: x[x != -1.0])
sampled_data = data.sample(n=10)

# Initialize variables to store depth and transmittance
depths = []
transmittances = []

# Process each sampled row
for _, row in sampled_data.iterrows():
    row = row.dropna().tolist()  # Drop NaN values, convert to list, and skip the first two elements
    depth = []
    transmittance = []
    T = 1.0  # Initialize the first transmittance value
    
    for i in range(1, len(row), 2):
        alpha = row[i-1]
        d = row[i]
        
        # Compute new transmittance
        T = T * (1 - alpha)
        
        # Append depth and transmittance to lists
        depth.append(d)
        transmittance.append(T)
    
    depths.append(depth)
    transmittances.append(transmittance)

# Plot the data for each sampled row
plt.figure(figsize=(12, 8))
for i in range(len(depths)):
    plt.step(depths[i], transmittances[i], where='post', marker='o', label=f'Pixel Sample {i+1}')
plt.xlabel('Depth')
plt.ylabel('Transmittance')
plt.title('Transmittance vs. Depth (Randomly Sampled Pixels)')
plt.legend()
plt.grid(True)
plt.savefig('/home/jiexiao/gaussian-splatting/analysis/transmittance_depth.png')


# Function to compute transmittance
def compute_transmittance(alphas):
    T = 1
    transmittance_values = []
    for alpha in alphas:
        T *= (1 - alpha)
        transmittance_values.append(T)
        if T < 0.0001:
            break
    return transmittance_values

# Plotting the transmittance
plt.figure(figsize=(12, 6))
for i in range(len(sampled_data)):
    # get alpha for odd index
    alphas = sampled_data.iloc[i].iloc[::2]
    transmittance_values = compute_transmittance(alphas)
    plt.plot(transmittance_values, label=f'Pixel {i+1}')

plt.title('Transmittance for each pixel')
plt.xlabel('Index')
plt.ylabel('Transmittance')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/jiexiao/gaussian-splatting/analysis/transmittance_vals.png')