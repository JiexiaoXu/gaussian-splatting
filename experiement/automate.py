import os
import subprocess
import shutil

views = [20, 50, 100]

analysis_dir = "/content/drive/MyDrive/3dgs/analysis"
model_dir = "/content/drive/MyDrive/3dgs/data/pre-trained/3DGS"
dataset_dir = "/content/drive/MyDrive/3dgs/data/tandt"
home_dir = "/content/drive/MyDrive/3dgs/gaussian-splatting"

analysis_names = [
    name
    for name in os.listdir(analysis_dir)
    if os.path.isdir(os.path.join(analysis_dir, name))
]
dataset_names = [
    name
    for name in os.listdir(dataset_dir)
    if os.path.isdir(os.path.join(dataset_dir, name))
]

# Generate csv file for each model at different view
for name in dataset_names:
    print(name)
    model_path = os.path.join(model_dir, name)
    dataset_path = os.path.join(dataset_dir, name)
    print(model_path + " " + dataset_path)
    for view in views:
        result_path = os.path.join(analysis_dir, name)
        result_path = os.path.join(result_path, "view" + str(view))
        if os.path.exists(os.path.join(result_path, "alpha_vals.csv")):
            continue
        command = [
            "python",
            "/content/drive/MyDrive/3dgs/gaussian-splatting/render_ray.py",
            "-m",
            model_path,
            "-s",
            dataset_path,
            "--view_index",
            str(view),
            "--skip_test",
        ]
        result = subprocess.run(command)

        print(result_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        shutil.move("/content/drive/MyDrive/3dgs/alpha_vals.csv", result_path)

# compute the data in each file


# Change the epsilon and other
