import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WIDTH = 980
HEIGHT = 545

# Read the CSV file
file_path = '/content/drive/MyDrive/3dgs/analysis/model01/view00/view00.csv'
data = pd.read_csv(file_path, header=None, delimiter=' ')

data = data.drop(columns=[0,1])

data = data.apply(lambda x: x[x != -1.0])

# print size of the data
print(data.shape)

def count_clusters_by_depth (data):
    '''find the number of clusters at same depth for each row in the data'''
    counts = []
    min_dist = []
    for _, row in data.iterrows():
        count = 0
        depth = 0.0
        distances = []
        row = row.dropna().tolist()  # Drop NaN values, convert to list, and skip the first two elements
        for i in range(1, len(row), 2):
            if row[i] - depth:
              distances.append(row[i] - depth)
              count += 1
              depth = row[i]  
        counts.append(count)
        min_dist.append(min(distances))
    return counts, min_dist

counts, min_dist = count_clusters_by_depth(data)
# print mode max min average and 99 percentile
# print(f"Mode: {max(set(counts), key=counts.count)}")
# print(f"99th percentile: {np.percentile(counts, 99)}")
# print(f"Max: {max(counts)}")
# print(f"Min: {min(counts)}")
# print(f"Average: {np.mean(counts)}")

# # draw histogram of counts 
# plt.hist(counts, bins=range(1, 15), align='left', rwidth=0.8)
# plt.xlabel('Number of clusters')
# plt.ylabel('Frequency')
# plt.title('Number of clusters at same depth')
# plt.grid(True)
# plt.savefig("")

percentile = 1

print(f"Mode: {max(set(min_dist), key=min_dist.count)}")
print(f"{percentile}th percentile: {np.percentile(min_dist, percentile)}")
print(f"Max: {max(min_dist)}")
print(f"Min: {min(min_dist)}")
print(f"Average: {np.mean(min_dist)}")

plt.hist(min_dist, bins=range(1, 15), align='left', rwidth=0.8)
plt.xlabel('Minimum distance between clusters')
plt.ylabel('Frequency')
plt.title('Minimum distance between clusters')
plt.grid(True)
plt.savefig("min_dist.png")
