import os
import shutil
import random

source_dir = "asl_alphabet_train"
target_base = "training/dataset"

classes = ["A", "B", "C", "N", "Y", "I"]

for cls in classes:
    images = os.listdir(os.path.join(source_dir, cls))
    random.shuffle(images)

    train_split = images[:300]
    val_split = images[300:350]
    test_split = images[350:400]

    for img in train_split:
        shutil.copy(
            os.path.join(source_dir, cls, img),
            os.path.join(target_base, "train", cls, img)
        )

    for img in val_split:
        shutil.copy(
            os.path.join(source_dir, cls, img),
            os.path.join(target_base, "val", cls, img)
        )

    for img in test_split:
        shutil.copy(
            os.path.join(source_dir, cls, img),
            os.path.join(target_base, "test", cls, img)
        )

print("Dataset prepared successfully.")
