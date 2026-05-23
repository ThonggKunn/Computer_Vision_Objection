# đã run
import os
import shutil

VOC2007 = "dataset/VOCdevkit/VOC2007"
VOC2012 = "dataset/VOCdevkit/VOC2012"

OUTPUT = "voc0712"

folders = [
    "JPEGImages",
    "Annotations",
    "ImageSets/Main"
]

for folder in folders:

    os.makedirs(os.path.join(OUTPUT, folder), exist_ok=True)

    for dataset in [VOC2007, VOC2012]:

        src = os.path.join(dataset, folder)

        if not os.path.exists(src):
            continue

        for file in os.listdir(src):

            src_file = os.path.join(src, file)
            dst_file = os.path.join(OUTPUT, folder, file)

            if os.path.isfile(src_file):
                shutil.copy(src_file, dst_file)

print("Merged VOC2007 + VOC2012 successfully")