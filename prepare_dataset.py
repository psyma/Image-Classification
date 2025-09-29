import os
import random
import shutil
from pathlib import Path

# datasets structure must follow this format
# root/dataset/train/class_x/xxx.png
# root/dataset/train/class_y/yyy.png
# root/dataset/val/class_x/xxx.png
# root/dataset/val/class_y/yyy.png

random.seed(42)
datasets = Path("./datasets/animals10/raw-img")
if os.path.exists("./datasets/dataset"):
    shutil.rmtree("./datasets/dataset")

for dataset in datasets.iterdir():
    os.makedirs(f"./datasets/dataset/train/{dataset.name}", exist_ok=True)
    os.makedirs(f"./datasets/dataset/val/{dataset.name}", exist_ok=True)
    os.makedirs(f"./datasets/dataset/test/{dataset.name}", exist_ok=True)
    
    files = list(dataset.iterdir()) 
    random.shuffle(files) 

    n_total = len(files)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.2)
    n_test = n_total - n_train - n_val  # ensure all files are used

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    # Create destination folders 
    train_path = Path(f"./datasets/dataset/train/{dataset.name}")
    val_path = Path(f"./datasets/dataset/val/{dataset.name}")
    test_path = Path(f"./datasets/dataset/test/{dataset.name}")

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    # Move or copy files
    for file in train_files:
        shutil.copy(file, train_path / file.name)

    for file in val_files:
        shutil.copy(file, val_path / file.name)

    for file in test_files:
        shutil.copy(file, test_path / file.name)

    print(f"{dataset.name}: {n_train} train, {n_val} val, {n_test} test files")