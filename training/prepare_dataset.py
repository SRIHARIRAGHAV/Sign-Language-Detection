import os
import shutil
import random

# Get absolute project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

source_dir = os.path.join(BASE_DIR, "asl_alphabet_train/asl_alphabet_train")
target_base = os.path.join(BASE_DIR, "training", "dataset")

classes = ["A", "B", "C", "N", "Y", "I"]

for cls in classes:
    src_path = os.path.join(source_dir, cls)

    if not os.path.exists(src_path):
        print("Folder not found:", src_path)
        continue

    images = os.listdir(src_path)
    random.shuffle(images)

    train_split = images[:300]
    val_split = images[300:350]
    test_split = images[350:400]

    for img in train_split:
        shutil.copy(
            os.path.join(src_path, img),
            os.path.join(target_base, "train", cls, img)
        )

    for img in val_split:
        shutil.copy(
            os.path.join(src_path, img),
            os.path.join(target_base, "val", cls, img)
        )

    for img in test_split:
        shutil.copy(
            os.path.join(src_path, img),
            os.path.join(target_base, "test", cls, img)
        )

print("Dataset prepared successfully.")
