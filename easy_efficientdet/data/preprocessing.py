import os
from typing import Any, Dict, Sequence, Union

import tensorflow as tf

from easy_efficientdet.boxencoding import BoxEncoder
from easy_efficientdet.config import ObjectDetectionConfig
from easy_efficientdet.data.augmentation import augment_data_builder
from easy_efficientdet.utils import get_tfds_size, setup_default_logger

# set up logging
logger = setup_default_logger("preprocessing")

# ensure tf version compatability
TFDATA_AUTOTUNE = tf.data.AUTOTUNE if hasattr(
    tf.data, "AUTOTUNE") else tf.data.experimental.AUTOTUNE

# TODO: are the custom properties bad for compatibility with tf object detection api ?

FEATURE_MAP = {
    "image/encoded": tf.io.VarLenFeature(tf.string),
    # convert to "image/encoded": tf.io.FixedLenFeature([], tf.string)
    # (and code after)
    "image/source_id": tf.io.VarLenFeature(tf.string),
    # convert to "image/source_id": tf.io.VarLenFeature([], tf.string),
    "image/height": tf.io.FixedLenFeature([], tf.int64),
    "image/width": tf.io.FixedLenFeature([], tf.int64),
    "image/filename": tf.io.FixedLenFeature([], tf.string),  # custom property
    "image/path": tf.io.FixedLenFeature([], tf.string),  # custom property
    "image/format": tf.io.FixedLenFeature([], tf.string),  # cusotm property
    "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
    # custom property
    "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    # custom property
    "image/object/class/text": tf.io.VarLenFeature(tf.string),
    "image/object/class/label": tf.io.VarLenFeature(tf.int64),
}


def init_data(config: ObjectDetectionConfig, mode: str = "train") -> tf.data.Dataset:

    if mode not in ("train", "val"):
        raise ValueError(f"'mode' parameter value must be 'train' or 'val' not {mode}")

    if mode == "train":
        return default_training_preprocessig(config)
    elif mode == "val":
        return default_validation_preprocessing(config)


def parse_od_record(raw_example: tf.Tensor) -> Dict[str, Union[str, tf.Tensor]]:
    """
    convert tf.train.Example instance into an image and respective
    labels
    """
    example = tf.io.parse_example(raw_example, features=FEATURE_MAP)

    if example["image/format"] == "png":
        image = tf.io.decode_png(
            tf.reshape(example["image/encoded"].values,
                       []))  # removed explicit channel dimension argument
    else:  # find out if this slows down the code
        image = tf.io.decode_jpeg(tf.reshape(example["image/encoded"].values, []))

    image_id = example["image/source_id"].values
    image_width = example["image/width"]
    image_height = example["image/height"]

    # decode bboxes and labels
    box_cls = example["image/object/class/label"].values
    box_cls_names = example["image/object/class/text"].values

    xmins = example["image/object/bbox/xmin"].values
    xmaxs = example["image/object/bbox/xmax"].values
    ymins = example["image/object/bbox/ymin"].values
    ymaxs = example["image/object/bbox/ymax"].values

    bboxes = tf.stack((ymins, xmins, ymaxs, xmaxs), axis=-1)
    filename = example["image/filename"]

    return {
        "source_id": image_id,
        "filename": filename,
        "image": image,
        "width": image_width,
        "height": image_height,
        "bbox_cls_names": box_cls_names,
        "bbox_cls": box_cls,
        "bboxes": bboxes,
        "bbbox_format": "rel_corner_yx",
    }


def load_tfrecords(path: str, tfrecord_suffix: str = "tfrecord") -> tf.data.Dataset:
    """Loads all tfrecord files contained path directory.

    Args:
        path (str): Path where to look for tfrecords

    Returns:
        [tf.data.TFRecordDataset]: tf-dataset containg all data contained in tfrecods,
            that are in the directory under path.
    """
    files = list(map(lambda file: os.path.join(path, file), os.listdir(path)))
    # filter out relevant files ending with .tfrecord
    tfrecord_files = list(
        filter(lambda file: file.endswith(tfrecord_suffix) and os.path.isfile(file),
               files))
    logger.info("Using {num} tfrecords".format(num=len(tfrecord_files)))

    display_files_amnt = 5
    log_files_text = "Creating tf-dataset from following tfrecord-files:\n{}".format(
        ", ".join(tfrecord_files[:display_files_amnt]))
    if len(tfrecord_files) > display_files_amnt:
        log_files_text += \
            f", and {len(tfrecord_files) - display_files_amnt} more file(s)"
    logger.info(log_files_text)
    # create tfdataset
    dataset = tf.data.TFRecordDataset(tfrecord_files)

    return dataset


def tfds_train_test_split(
    tfds: tf.data.Dataset,
    test_frac: float,
    dataset_size: Union[int, str],
    buffer_size: int = 256,
    seed: int = 123,
) -> Sequence[Union[tf.data.Dataset, tf.data.Dataset, int, int]]:
    """
    !!! does not properly work, seems to be dependant on hardware, open isssue on
    github/tensorflow?

    Split tf-dataset into a train and test dataset.
    https://stackoverflow.com/questions/48213766/split-a-dataset-created-by-tensorflow-dataset-api-in-to-train-and-test

    Args:
        tfds (tf.data.Dataset): Tf-dataset, that will be split into a train- and
            testset.
        test_frac (float): Fract

    Returns:
        [tf.data.Dataset, tf.data.Dataset, int, int]: Returns train and test datasets
            as well as the absolut sizes of the full and the train dataset.
    """
    logger.warning(
        "This methods of data splitting does not gurantee same split on every machine.")
    full_ds_size = None

    if dataset_size == "auto":
        logger.warning(
            "dataset_size='auto': In order to calculate the size of the dataset, all "
            "samples will be loaded.")
        full_ds_size = get_tfds_size(tfds)
    elif isinstance(dataset_size, int):
        full_ds_size = dataset_size

    logger.info(f"Using following seed to shuffle data: {seed}")
    tfds = tfds.shuffle(buffer_size, reshuffle_each_iteration=False, seed=seed)

    train_ds_fraction = 1.0 - test_frac
    train_ds_size = int(train_ds_fraction * full_ds_size)
    logger.info(f"train dataset size: {train_ds_size}, val dataset size: "
                "{full_ds_size - train_ds_size}")
    train_dataset = tfds.take(train_ds_size)
    test_dataset = tfds.skip(train_ds_size)

    return train_dataset, test_dataset, full_ds_size, train_ds_size


def extract_training_properties(sample: Dict[str, Any]):
    return {
        "image": sample["image"],
        "bboxes": sample["bboxes"],
        "labels": sample["bbox_cls"],
    }


def default_training_preprocessig(config: ObjectDetectionConfig) -> tf.data.Dataset:

    augment_data = augment_data_builder(**config.get_augmentation_config())
    encoder = BoxEncoder(**config.get_encoding_config())

    data = load_tfrecords(config.train_data_path, config.tfrecord_suffix)
    # dataset_size = get_tfds_size(data)
    # print('data set size', dataset_size) # change this to logging
    data = data.map(parse_od_record, TFDATA_AUTOTUNE)
    data = data.map(extract_training_properties, TFDATA_AUTOTUNE)
    # TODO snapshots would make sense here, would work with tf.__version___ >= 2.6.0

    # TODO add buffer size to config (low priority)
    data = data.shuffle(256)
    data = data.map(augment_data, TFDATA_AUTOTUNE)
    data = data.map(lambda *sample: encoder.encode(sample[0], sample[1], sample[2]),
                    TFDATA_AUTOTUNE)  # noqa
    data = data.batch(batch_size=config.batch_size, drop_remainder=True)
    data = data.prefetch(TFDATA_AUTOTUNE)

    return data


def _val_preprocess_builder(image_shape: Sequence[int]):
    def val_preprocess(sample: Dict[str, Any]):
        image = tf.cast(sample['image'], tf.float32)
        image = tf.image.resize(image, image_shape[:2])
        return (image, sample['bboxes'], sample['labels'])

    return val_preprocess


def default_validation_preprocessing(config: ObjectDetectionConfig) -> tf.data.Dataset:

    encoder = BoxEncoder(**config.get_encoding_config())

    data_test = load_tfrecords(config.val_data_path, config.tfrecord_suffix)
    data_test = data_test.map(parse_od_record, TFDATA_AUTOTUNE)
    data_test = data_test.map(extract_training_properties, TFDATA_AUTOTUNE)

    val_preprocessing = _val_preprocess_builder(config.image_shape)

    data_test = data_test.map(val_preprocessing, TFDATA_AUTOTUNE)
    data_test = data_test.map(
        lambda *sample: encoder.encode(sample[0], sample[1], sample[2]),
        TFDATA_AUTOTUNE)
    # TODO add snapshots here for tf >= 2.6.0

    data_test = data_test.batch(batch_size=config.batch_size, drop_remainder=True)
    data_test = data_test.prefetch(TFDATA_AUTOTUNE)

    return data_test
