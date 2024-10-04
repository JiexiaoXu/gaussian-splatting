import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
from PIL import Image, ImageDraw
from itertools import chain


def collect_min_dist(data, epsilon):
    """Find the number of clusters at the same depth for each row in the data"""
    dist_pix = dict()
    counts = []
    for idx, (_, row) in enumerate(data.iterrows()):
        pixel_dist = []
        depth = 0.0
        row = row.dropna().tolist()
        count = 0

        for i in range(1, len(row), 2):
            if row[i] - depth > epsilon:
                pixel_dist.append(row[i] - depth)
                depth = row[i]
                count += 1

        # Skip if only one cluster exists
        counts.append(count)
        if count == 0:
            continue

        pix_min_dist = min(pixel_dist)
        if pix_min_dist not in dist_pix:
            dist_pix[pix_min_dist] = [idx]
        else:
            dist_pix[pix_min_dist].append(idx)
    return counts, dist_pix


def circle_pixel(image, file_name, target_pix, width):
    """Find the pixels corresponding to the first 10 minimum distances in the image"""

    # Get the pixels corresponding to the first 10 minimum distances
    draw = ImageDraw.Draw(image)
    for pix in target_pix:
        x = int(pix % width)
        y = int(pix // width)
        print(f"Pixel coordinates for minimum distance {pix}: ({x}, {y})")

        # circle the pixels in the image
        # Add the name near the circle
        radius = 8  # Radius of the circle to highlight the pixel
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius), outline="red", width=2
        )

    # save the image
    image.save(
        f"/content/drive/MyDrive/3dgs/analysis/cluster_eval/{file_name}_min_gap.png"
    )


def plot_transmittance_depth(data, target_pix, file_name):
    transmittances = dict()
    depths = dict()

    for pix in target_pix:
        row = data.iloc[pix]
        row = row.dropna().tolist()
        depth = []
        transmittance = []
        T = 1.0  # Initialize the first transmittance value

        for i in range(1, len(row), 2):
            alpha = row[i - 1]
            d = row[i]

            # Compute new transmittance
            T = T * (1 - alpha)

            # Append depth and transmittance to lists
            depth.append(d)
            transmittance.append(T)

        depths[pix] = depth
        transmittances[pix] = transmittance

    plt.figure(figsize=(12, 8))
    for pix in target_pix:
        plt.step(
            depths[pix],
            transmittances[pix],
            where="post",
            marker="o",
            label=f"Pixel {pix}",
        )

    plt.xlabel("Depth")
    plt.ylabel("Transmittance")
    plt.title("Transmittance vs. Depth (Randomly Sampled Pixels)")
    plt.legend()
    plt.grid(True)

    fig_name = file_name + "_cluster"
    fig_path = f"/content/drive/MyDrive/3dgs/analysis/cluster_eval/{fig_name}.png"
    plt.savefig(fig_path)
    plt.show()


def plot_heatmap(counts, height, width, file_name):
    print(f"count size: {len(counts)}, expected size: {height * width}")
    np_counts = np.array(counts)
    count_2d = np_counts.reshape((height, width))

    plt.figure()
    plt.imshow(count_2d, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Depth Complexity")
    plt.title("Heatmap of Depth Complexity per Pixel")

    heat_name = file_name + "_heat"
    heat_path = f"/content/drive/MyDrive/3dgs/analysis/cluster_eval/{heat_name}.png"
    plt.savefig(heat_path)
    plt.show()


if __name__ == "__main__":
    # Parse the arguments and take the file path and image path
    parser = argparse.ArgumentParser(
        description="Find the minimum gap between clusters"
    )
    parser.add_argument("file_path", type=str, help="Path to the CSV file")
    parser.add_argument("image_path", type=str, help="Path to the image file")

    parser.add_argument(
        "--sample_num", type=int, default=2, help="Number of pixels to sample"
    )
    parser.add_argument(
        "--epsilon", type=float, default=5e-6, help="Number of pixels to sample"
    )

    args = parser.parse_args()
    file_path = args.file_path
    image_path = args.image_path
    sample_num = args.sample_num
    epsilon = args.epsilon

    # Read the data from the CSV file
    file_name = file_path.split("/")[-2] + "_" + file_path.split("/")[-3]
    data = pd.read_csv(file_path, header=None, delimiter=" ")
    data = data.drop(columns=[0, 1])
    data = data.apply(lambda x: x[x != -1.0])

    # Find the minimum gap between clusters, process the data
    counts, dist_pix = collect_min_dist(data, epsilon)
    sorted_dist = sorted(dist_pix.keys())
    target = sorted_dist[:sample_num]

    # Read the image
    image = Image.open(image_path)
    width, height = image.size

    # Find corresponding row numbers
    target_pix = list(chain.from_iterable(dist_pix[key] for key in target))

    # Plot the graphs (circles, transmittance, heatmap)
    circle_pixel(image, file_name, target_pix, width)
    plot_transmittance_depth(data, target_pix, file_name)
    plot_heatmap(counts, height, width, file_name)
