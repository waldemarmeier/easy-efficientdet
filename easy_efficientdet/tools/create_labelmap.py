import argparse
import json
import logging
import pathlib
import sys
import xml.etree.ElementTree as ET

logger = logging.getLogger("create-labelmap")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def create_labelmap(data_dir: str, labelmap_file: str):

    object_name_set = set()

    for xml_file in data_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        object_names = root.findall("object/name")
        object_name_set.update(map(lambda x: x.text, object_names))

    object_names = sorted(object_name_set)
    logger.info(f"found following objects: \n{object_names}")

    out_map = [
        dict(id=idx, name=object_name)
        for idx, object_name in enumerate(object_names, start=1)
    ]

    with open(labelmap_file, "w") as fs:
        json.dump(out_map, fs, indent=4)


def main():

    parser = argparse.ArgumentParser(
        description="Create label map from Pascal/VOC type dataset.")
    parser.add_argument(
        "-i",
        "--input-dir",
        help="directory containing containing annotations in XML-files",
        type=str,
    )
    parser.add_argument("-o",
                        "--output-file",
                        help="output text file containg the labelmap",
                        type=str)
    parser.add_argument("-ow", "--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.input_dir)
    labelmap_file = args.output_file

    if (not args.overwrite) and pathlib.Path(labelmap_file).is_file():
        logger.info(f"file {labelmap_file} alredy exists! Set --overwrite " +
                    "flag for the file to be overwritten")
        exit()

    create_labelmap(data_dir, labelmap_file)


if __name__ == "__main__":
    main()
