import argparse
import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

import dataset_util
import tensorflow as tf
from dataset_util import (
    decode_image,
    dict_chunks,
    get_auto_shard_size,
    get_validate_size,
    label_map_to_dict,
    merge_annotatations,
    parse_source_tag,
    validate_label_map,
)

logger = logging.getLogger("generate-tfrecord")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

DEFAULT_TFRECORD_NAME = "data{shardnum}.tfrecord"
_SHARD_SIZE_MB = 100


def read_label_map(path: str) -> dict:
    # load label map json
    with open(path) as fs:
        label_map_json = json.load(fs)

    validate_label_map(label_map_json)

    label_map_dict = label_map_to_dict(label_map_json)

    return label_map_dict


def parse_bboxes(bndboxes: List[ET.Element]):

    parsed_bboxes = []

    for bbox in bndboxes:
        parsed_bboxes.append({
            "class": bbox.find("name").text,
            "xmax": int(bbox.find("bndbox/xmax").text),
            "xmin": int(bbox.find("bndbox/xmin").text),
            "ymax": int(bbox.find("bndbox/ymax").text),
            "ymin": int(bbox.find("bndbox/ymin").text),
        })

    return parsed_bboxes


def parse_pascal_xml(path: str) -> Dict[str, Any]:

    tree = ET.parse(path)
    root = tree.getroot()
    filename = root.find("filename").text

    if filename is None:
        raise Exception(f"Empty filename for {path}")

    if root.find("path") is None:
        path = None
    else:
        path = root.find("path").text

    # parse source tag can be None
    source = parse_source_tag(root.find("source"))
    height = int(root.find("size/height").text)
    width = int(root.find("size/width").text)

    if (width is None) or (height is None):
        raise Exception(f"Width or height for annotation file {path} is not present")

    bboxes = parse_bboxes(root.findall("object"))

    res_anno = dict()
    res_anno["filename"] = filename
    if path is not None:
        res_anno["path"] = path
    if source is not None:
        res_anno["source"] = source

    res_anno["bboxes"] = bboxes
    res_anno["width"] = width
    res_anno["height"] = height

    return res_anno


def load_annotations(data_dir: str) -> dict:
    annotations = {}

    for ann_file in Path(data_dir).glob("*.xml"):

        # load annotations in memory
        parsed_anno = parse_pascal_xml(ann_file)
        new_filename = parsed_anno["filename"]

        if new_filename in annotations:
            logging.warning(f"merging annotations with same filename {new_filename}")
            merged_anno = merge_annotatations(
                new_filename,
                annotations[new_filename],
                parsed_anno,
            )
            annotations[new_filename] = merged_anno
        else:
            annotations[new_filename] = parsed_anno

    return annotations


def create_tfrecords(
    input_dir: str,
    output_dir: str,
    path_labelmap: str,
    out_file_template: str = None,
    shards: int = 1,
) -> None:

    if shards < 1:
        raise Exception(f"invalid shard number {shards}")

    label_map = read_label_map(path_labelmap)

    annotations = load_annotations(input_dir)
    out_file_template = (DEFAULT_TFRECORD_NAME
                         if out_file_template is None else out_file_template)

    if ("shardnum" not in out_file_template) and shards > 1:
        out_file_template + "-{shardnum}"
    # out_file_template = out_file_template if ('shardnum' not in out_file_template)
    # and shards > 1 else out_file_template + '-{shardnum}'

    out_file_template = os.path.join(output_dir, out_file_template)
    anno_per_shard = int(len(annotations) / shards) + 1

    for idx, anno_shard in enumerate(dict_chunks(annotations, anno_per_shard)):
        idx += 1
        out_file = out_file_template.format(shardnum=f"{idx:05d}-{shards:05d}")
        logger.info(f"creating tfrecord: {out_file}")
        with tf.io.TFRecordWriter(out_file) as w:

            for anno in anno_shard.values():
                # read image and encode it
                image_filename = anno["filename"]
                image_filename_encoded = image_filename.encode("utf-8")
                image_encoded = tf.io.read_file(os.path.join(input_dir,
                                                             image_filename)).numpy()
                image, image_format = decode_image(image_encoded, image_filename)
                image_format_encoded = image_format.encode("utf-8")
                height, width = get_validate_size(image, anno, image_filename)

                if "source" in anno:
                    source_encoded = anno["source"].encode("utf-8")
                else:
                    source_encoded = b""

                if "path" in anno:
                    path_encoded = anno["path"].encode("utf-8")
                else:
                    path_encoded = b""

                xmins = [bbox["xmin"] / width for bbox in anno["bboxes"]]
                xmaxs = [bbox["xmax"] / width for bbox in anno["bboxes"]]
                ymins = [bbox["ymin"] / height for bbox in anno["bboxes"]]
                ymaxs = [bbox["ymax"] / height for bbox in anno["bboxes"]]
                classes_text_encoded = [
                    bbox["class"].encode("utf-8") for bbox in anno["bboxes"]
                ]
                classes = [label_map[bbox["class"]] for bbox in anno["bboxes"]]

                tf_example = tf.train.Example(features=tf.train.Features(
                    feature={
                        "image/height":
                        dataset_util.int64_feature(height),
                        "image/width":
                        dataset_util.int64_feature(width),
                        "image/filename":
                        dataset_util.bytes_feature(image_filename_encoded),
                        "image/source_id":
                        dataset_util.bytes_feature(source_encoded),
                        "image/path":
                        dataset_util.bytes_feature(path_encoded),
                        "image/encoded":
                        dataset_util.bytes_feature(image_encoded),
                        "image/format":
                        dataset_util.bytes_feature(image_format_encoded),
                        "image/object/bbox/xmin":
                        dataset_util.float_list_feature(xmins),
                        "image/object/bbox/xmax":
                        dataset_util.float_list_feature(xmaxs),
                        "image/object/bbox/ymin":
                        dataset_util.float_list_feature(ymins),
                        "image/object/bbox/ymax":
                        dataset_util.float_list_feature(ymaxs),
                        "image/object/class/text":
                        dataset_util.bytes_list_feature(classes_text_encoded),
                        "image/object/class/label":
                        dataset_util.int64_list_feature(classes),
                    }))

                w.write(tf_example.SerializeToString())


def main():

    parser = argparse.ArgumentParser(
        description="Create directory containing Pascal/VOC train/val/"
        "trainval images and respective annotations from the standard "
        "directory structure of the Pascal/VOC dataset.")

    parser.add_argument(
        "-i",
        "--input-dir",
        help="directory containing with image and xml files",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="output directory for tfrecords",
        type=str,
        required=True,
    )
    parser.add_argument("-l",
                        "--label-map",
                        help="path to label map",
                        type=str,
                        required=True)
    parser.add_argument("-f",
                        "--file-template",
                        help="output directory for tfrecords",
                        type=str)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-s",
        "--shards",
        help="Number of output data shards, if 1, all data will written into one "
        "tf-record",
        type=int,
    )
    group.add_argument(
        "-as",
        "--auto-shard",
        help="Number of output data shards is determined automatically so that "
        "resulting tfrecods have a size of 100-199 MB",
        action="store_true",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    path_labelmap = args.label_map
    out_file_template = (DEFAULT_TFRECORD_NAME
                         if args.file_template is None else args.file_template)

    num_shards = 1  # default, put everything into one tf-record

    if args.auto_shard:
        num_shards = get_auto_shard_size(input_dir, _SHARD_SIZE_MB)
    elif args.shards is not None:
        num_shards = args.shards

    create_tfrecords(
        input_dir=input_dir,
        output_dir=output_dir,
        path_labelmap=path_labelmap,
        out_file_template=out_file_template,
        shards=num_shards,
    )


if __name__ == "__main__":
    main()
