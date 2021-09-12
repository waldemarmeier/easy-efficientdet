import copy
import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from itertools import islice

import tensorflow
import tensorflow as tf

logger = logging.getLogger("dataset-util")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_validate_size(image: tf.Tensor, anno: dict, filename: str):

    height, width, _ = image.shape

    if not ((anno["width"] == width) and anno["height"] == height):
        raise Exception(
            f"{filename}:width or height in annotation does not match image")

    return height, width


def decode_image(encoded_image, filename: str) -> tf.Tensor:

    if filename.endswith("jpg") or filename.endswith("jpeg"):
        return tf.io.decode_jpeg(encoded_image), "jpg"
    elif filename.endswith("png"):
        return tf.io.decode_png(encoded_image), "png"
    else:
        raise Exception(f"Unknown image file type: {filename}")


def get_auto_shard_size(data_dir: str, file_size: int) -> int:
    # tries to split in shards of 100-199 MBs
    num_mgbytes = sum(d.stat().st_size
                      for d in os.scandir(data_dir) if d.is_file()) / (10**6)
    logger.info(f"directory size: {num_mgbytes}MB")
    num_shards = int(num_mgbytes / file_size)
    logger.info(f"creating {num_shards} shards")
    return num_shards


def dict_chunks(data, size) -> dict:
    # https://stackoverflow.com/questions/22878743/how-to-split-dictionary-into-multiple-dictionaries-fast
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


def parse_source_tag(source_tag: ET.Element):

    if source_tag is None:
        return None

    out_json = dict()
    for tag in ("annotation", "database", "image"):
        if source_tag.find(tag) is not None:
            out_json[tag] = source_tag.find(tag).text

    if len(out_json) > 0:
        return json.dumps(out_json)
    else:
        return None


def validate_label_map(label_map):

    known_ids = set()
    known_names = set()
    for label in label_map:
        if label["id"] <= 0:
            raise Exception(
                f"Id for label {label['name']} is 0 or below: {label['id']}")

        if not (label["name"] in known_names):
            known_names.add(label["name"])
        else:
            raise Exception(f"duplicate id: {label['name']}")

        if not (label["id"] in known_ids):
            known_ids.add(label["id"])
        else:
            raise Exception(f"duplicate name: {label['id']}")


def merge_annotatations(filename, anno1, anno2):

    # make consistency check
    if not (filename == anno1["filename"]):
        raise Exception(
            f"file name {anno1['filename']} in annotation does not match expected "
            "filename {filename}")
    if not (anno1["filename"] == anno2["filename"]):
        raise Exception(
            f"Inconsistencies in filenames for annotations: {anno1} and {anno2}")
    if not (anno1["source"] == anno2["source"]):
        raise Exception(f"inconsistent source name: {anno1} and {anno2}")
    # do not care about path
    if not ((anno1["height"] == anno2["height"]) and
            (anno1["width"] == anno2["width"])):
        raise Exception(
            f"image size doest not match for annotations: {anno1} and {anno2}")

    # compare annotations for duplicates
    for bbox1 in anno1["bboxes"]:
        for bbox2 in anno2["bboxes"]:
            if bbox1["class"] == bbox2["class"]:
                # check if annotation has same class and bounding box coordinates
                if (bbox1["xmax"] == bbox2["xmax"] and bbox1["xmin"] == bbox2["xmin"]
                        and bbox1["ymax"] == bbox2["ymax"]
                        and bbox1["ymin"] == bbox1["ymin"]):
                    raise Exception(
                        f"File:{filename} contains duplicates bounding boxes: {anno1} "
                        "{anno1}")
            else:
                continue

    new_anno = copy.deepcopy(anno1)
    new_bboxes = copy.deepcopy(anno2["bboxes"])
    new_anno["bboxes"].extend(new_bboxes)
    return new_anno


def label_map_to_dict(label_map):
    return {label["name"]: label["id"] for label in label_map}


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
