import unittest

import tensorflow as tf

from easy_efficientdet import DefaultConfig
from easy_efficientdet.utils import ImageColor, convert_image_to_rgb


class TestSum(unittest.TestCase):
    def test_simple_default_config(self):

        num_cls = 9
        batch_size = 32
        train_data_path = 'train_data'
        val_data_path = 'val_data'
        epochs = 8
        training_image_size = 512
        efficientdet_version = 0
        image_data_color = ImageColor.RGB
        config = DefaultConfig(num_cls=num_cls,
                               batch_size=batch_size,
                               train_data_path=train_data_path,
                               val_data_path=val_data_path,
                               epochs=epochs,
                               training_image_size=training_image_size,
                               efficientdet_version=efficientdet_version,
                               image_data_color=image_data_color)
        self.assertEqual(num_cls, config.num_cls)
        self.assertEqual(batch_size, config.batch_size)
        self.assertEqual(train_data_path, config.train_data_path)
        self.assertEqual(val_data_path, config.val_data_path)
        self.assertEqual(epochs, config.epochs)
        self.assertEqual(training_image_size, config.training_image_size)
        self.assertEqual(efficientdet_version, config.efficientdet_version)
        # if bw_image_data == True then tf.image.grayscale_to_rgb image preprocessor
        self.assertEqual(tf.identity, config.image_preprocessor)

    def test_bw_default_config(self):

        image_data_color = ImageColor.BW
        config = DefaultConfig(num_cls=9,
                               batch_size=32,
                               train_data_path='train_data_path',
                               val_data_path='val_data_path',
                               epochs=8,
                               training_image_size=512,
                               efficientdet_version=0,
                               image_data_color=image_data_color)
        # if bw_image_data == True then tf.image.grayscale_to_rgb image preprocessor
        self.assertEqual(tf.image.grayscale_to_rgb, config.image_preprocessor)

    def test_mixed_default_config(self):

        image_data_color = ImageColor.MIXED
        config = DefaultConfig(num_cls=9,
                               batch_size=32,
                               train_data_path='train_data_path',
                               val_data_path='val_data_path',
                               epochs=8,
                               training_image_size=512,
                               efficientdet_version=0,
                               image_data_color=image_data_color)
        # if bw_image_data == True then tf.image.grayscale_to_rgb image preprocessor
        self.assertEqual(convert_image_to_rgb, config.image_preprocessor)


if __name__ == '__main__':
    unittest.main()
