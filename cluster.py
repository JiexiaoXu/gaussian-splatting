import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import shutil

# Read the CSV file
analysis_dir = "/content/drive/MyDrive/3dgs/analysis"
tandt_dir = "/content/drive/MyDrive/3dgs/data/tandt"


def count_cluster(fname, view_name, epsilon, percentile):
    view_path = os.path.join(analysis_dir, fname, view_name)
    print(view_path)
    data = pd.read_csv(
        os.path.join(view_path, "alpha_vals.csv"), header=None, delimiter=" "
    )
    data = data.drop(columns=[0, 1])
    data = data.apply(lambda x: x[x != -1.0])

    def count_clusters_by_depth(data):
        """find the number of clusters at same depth for each row in the data"""
        counts = []
        min_dist = []
        for _, row in data.iterrows():
            count = 0
            depth = 0.0
            distances = []
            row = (
                row.dropna().tolist()
            )  # Drop NaN values, convert to list, and skip the first two elements
            for i in range(1, len(row), 2):
                if abs(row[i] - depth) >= epsilon:
                    distances.append(row[i] - depth)
                    count += 1
                    depth = row[i]
            if count == 0:
                continue
            counts.append(count)
            min_dist.append(min(distances))
        return counts, min_dist

    counts, min_dist = count_clusters_by_depth(data)

    data = {
        "Cluster# Mode": max(set(counts), key=counts.count),
        f"Cluster# {percentile}th Percentile": np.percentile(counts, percentile),
        "Cluster# Max": max(counts),
        "Cluster# Min": min(counts),
        "Cluster# Average": np.mean(counts),
        "Distance Mode": max(set(min_dist), key=min_dist.count),
        "Distance 1st Percentile": np.percentile(min_dist, 1),
        "Distance 5th Percentile": np.percentile(min_dist, 5),
        "Distance Max": max(min_dist),
        "Distance Min": min(min_dist),
        "Distance Average": np.mean(min_dist),
    }

    # Save to a JSON file
    stat_name = f"stats_{percentile}th_{epsilon}.json"
    stat_path = os.path.join(view_path, stat_name)
    with open(stat_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    # draw histogram of counts
    plt.hist(counts, bins=range(1, 20), align="left", rwidth=0.8)
    plt.xlabel("Number of clusters")
    plt.ylabel("Frequency")
    plt.title("Number of clusters at same depth")
    plt.grid(True)

    fig_path = os.path.join(view_path, f"cluster_{percentile}th_{epsilon}.png")
    plt.savefig(fig_path)

    # print(count_clusters_by_depth(data))
    # print(len(count_clusters_by_depth(data)))

    width = 979
    height = 546
    np_counts = np.array(counts)
    count_2d = np_counts.reshape((height, width))

    plt.imshow(count_2d, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Depth Complexity")
    plt.title("Heatmap of Depth Complexity per Pixel")

    heat_path = os.path.join(view_path, f"heat_{percentile}th_{epsilon}.png")
    plt.savefig(heat_path)


if __name__ == "__main__":
    analysis_names = [
        name
        for name in os.listdir(analysis_dir)
        if os.path.isdir(os.path.join(analysis_dir, name))
    ]

    percentiles = [95, 99]

    fname = "truck"
    views = [name for name in os.listdir(os.path.join(analysis_dir, fname))]
    for view in views:
        for percentile in percentiles:
            count_cluster(fname, view, 5e-6, percentile)
