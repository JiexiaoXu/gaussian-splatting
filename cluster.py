import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
file_path = '/content/drive/MyDrive/3dgs/analysis/model01/view00/view00.csv'
data = pd.read_csv(file_path, header=None, delimiter=' ')

data = data.drop(columns=[0,1])

data = data.apply(lambda x: x[x != -1.0])

print(data.shape)

def count_clusters_by_depth (data):
    '''find the number of clusters at same depth for each row in the data'''
    counts = []
    for _, row in data.iterrows():
        count = 0
        depth = 0.0
        row = row.dropna().tolist()  # Drop NaN values, convert to list, and skip the first two elements
        for i in range(1, len(row), 2):
            if abs(row[i] - depth)>= 0.01:
                count += 1
                depth = row[i]
        if count == 0:
          continue
        counts.append(count)
    return counts

counts = count_clusters_by_depth(data)
percentile = 95

# print mode max min average and 99 percentile
print(f"Mode: {max(set(counts), key=counts.count)}")
print(f"{percentile}th: {np.percentile(counts, percentile)}")
print(f"Max: {max(counts)}")
print(f"Min: {min(counts)}")
print(f"Average: {np.mean(counts)}")


# draw histogram of counts 
plt.hist(counts, bins=range(1, 20), align='left', rwidth=0.8)
plt.xlabel('Number of clusters')
plt.ylabel('Frequency')
plt.title('Number of clusters at same depth')
plt.grid(True)
plt.savefig('cluster_count.png')

# print(count_clusters_by_depth(data))
# print(len(count_clusters_by_depth(data)))

width = 980
height = 545
np_counts = np.array(counts)
count_2d = np_counts.reshape((height, width))

plt.imshow(count_2d, cmap='hot', interpolation='nearest')
plt.colorbar(label='Depth Complexity')
plt.title('Heatmap of Depth Complexity per Pixel')
plt.savefig('dc_heat.png')

