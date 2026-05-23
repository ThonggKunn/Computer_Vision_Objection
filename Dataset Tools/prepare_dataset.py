import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm

classes = [
'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair',
'cow','diningtable','dog','horse','motorbike','person','pottedplant',
'sheep','sofa','train','tvmonitor'
]

VOC_PATH = "../voc0712"
IMAGE_DIR = os.path.join(VOC_PATH, "JPEGImages")
ANNOT_DIR = os.path.join(VOC_PATH, "Annotations")
SPLIT_DIR = os.path.join(VOC_PATH, "ImageSets/Main")

OUTPUT_IMG = os.path.join(VOC_PATH, "images")
OUTPUT_LABEL = os.path.join(VOC_PATH, "labels")


def convert(size, box):

    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0

    w = box[1] - box[0]
    h = box[3] - box[2]

    x *= dw
    y *= dh
    w *= dw
    h *= dh

    return (x, y, w, h)


def convert_annotation(image_id, split):

    xml_path = os.path.join(ANNOT_DIR, image_id + ".xml")
    label_path = os.path.join(OUTPUT_LABEL, split, image_id + ".txt")

    if not os.path.exists(xml_path):
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")

    w = int(size.find("width").text)
    h = int(size.find("height").text)

    with open(label_path, "w") as out_file:

        for obj in root.iter("object"):

            cls = obj.find("name").text

            if cls not in classes:
                continue

            cls_id = classes.index(cls)

            xmlbox = obj.find("bndbox")

            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text)
            )

            bb = convert((w, h), b)

            out_file.write(str(cls_id) + " " + " ".join(map(str, bb)) + "\n")


splits = ["train", "val", "test"]

for split in splits:

    print("Processing", split)

    img_out = os.path.join(OUTPUT_IMG, split)
    label_out = os.path.join(OUTPUT_LABEL, split)

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)

    split_file = os.path.join(SPLIT_DIR, split + ".txt")

    with open(split_file) as f:
        ids = f.read().strip().split()

    for image_id in tqdm(ids):

        img_src = os.path.join(IMAGE_DIR, image_id + ".jpg")
        img_dst = os.path.join(img_out, image_id + ".jpg")

        if os.path.exists(img_src):
            shutil.copy(img_src, img_dst)

        convert_annotation(image_id, split)

print("VOC0712 → YOLO dataset ready")