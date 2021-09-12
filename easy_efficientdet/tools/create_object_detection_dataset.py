import argparse
import logging
import os
import shutil
import sys

logger = logging.getLogger("create-object-detection-dataset")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

IMAGE_SETS = {"train", "val", "trainval", "test"}


def create_dataset(output_dir: str, input_dir: str, image_set: str) -> None:

    os.makedirs(output_dir)
    logging.info(f"created output directory {output_dir}")

    if image_set not in IMAGE_SETS:
        raise Exception(f"Unkwown image set {image_set} not in {IMAGE_SETS}")

    # load file
    image_list = None
    iamge_list_file = os.path.join(input_dir, "ImageSets", "Main", f"{image_set}.txt")

    with open(iamge_list_file) as fs:
        image_list = list(map(lambda x: x.replace("\n", ""), fs.readlines()))

    logger.info(f"using image list file: {iamge_list_file}")

    i = 0

    for img_id in image_list:

        img_file_in = os.path.join(input_dir, "JPEGImages", f"{img_id}.jpg")
        ann_file_in = os.path.join(input_dir, "Annotations", f"{img_id}.xml")

        img_file_out = os.path.join(output_dir, f"{img_id}.jpg")
        ann_file_out = os.path.join(output_dir, f"{img_id}.xml")

        shutil.copy(img_file_in, img_file_out)
        shutil.copy(ann_file_in, ann_file_out)

        i += 1

    logger.info(f"Moved {i} images and annotations from {input_dir} to {output_dir}")


def main():

    parser = argparse.ArgumentParser(
        description="Create directory containing Pascal/VOC train/val/"
        "trainval images and respective annotations from the standard "
        "directory structure of the Pascal/VOC dataset.")

    parser.add_argument(
        "-i",
        "--input-dir",
        help="directory containing ImageSets/Main/, JPEGImages and Annotaions "
        "directories",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="output directory for annotations and respective images",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--image-set",
        help="image set to be moved, element in train, val, trainval or test",
        choices=("train", "val", "trainval", "test"),
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    input_dir = args.input_dir
    image_set = args.image_set

    create_dataset(output_dir, input_dir, image_set)


if __name__ == "__main__":
    main()
