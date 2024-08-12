import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WIDTH = 980
HEIGHT = 545

# Read the CSV file
file_path = 'alpha_vals.csv'
data = pd.read_csv(file_path, header=None, delimiter=' ')

data = data.drop(columns=[0,1])

data = data.apply(lambda x: x[x != -1.0])

print(data.tail())

def count_splats_by_pixels(data):
    '''find the number of splats for each pixel in the data'''
    counts = []
    for _, row in data.iterrows():
        count = 0
        row = row.dropna().tolist()  # Drop NaN values, convert to list, and skip the first two elements
        for i in range(1, len(row), 2):
            count += row[i+1]
        counts.append(count)

counts = count_splats_by_pixels(data)
# print mode max min average and 99 percentile
print(f"Mode: {max(set(counts), key=counts.count)}")
print(f"99th percentile: {np.percentile(counts, 99)}")
print(f"Max: {max(counts)}")
print(f"Min: {min(counts)}")
print(f"Average: {np.mean(counts)}")


# draw histogram of counts 
plt.hist(counts, bins=range(1, 15), align='left', rwidth=0.8)
plt.xlabel('Number of clusters')
plt.ylabel('Frequency')
plt.title('Number of clusters at same depth')
plt.grid(True)
plt.savefig('/home/jiexiao/gaussian-splatting/analysis/cluster_count.png')
plt.show()

print(count_splats_by_pixels(data))
print(len(count_splats_by_pixels(data)))

