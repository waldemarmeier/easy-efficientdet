from setuptools import find_packages, setup

# TODO major todo
# integrate scripts with entry point
# https://stackoverflow.com/questions/774824/explain-python-entry-points

requirements = {"default": ["tensorflow>=2.3.0"], "optional": {"": ""}}


def load_readme() -> str:
    with open('README.md') as fp:
        return fp.read()


setup(name='easy-efficientdet',
      version='0.1',
      author="Waldemar Meier",
      author_email="info@waldemarmeier.com",
      url="https://github.com/waldemarmeier/easy-efficientdet",
      keywords="object-detection neural-network efficientdet",
      packages=find_packages(exclude=["easy_efficientdet._third_party*"]),
      license='Apache 2.0',
      description="Easy to use object detection package based on tensorflow",
      long_description=load_readme(),
      entry_points={
          'console_scripts': [
              'generate_tfrecord = easy_efficientdet.tools.generate_tfrecord:main',
              'create_labelmap = easy_efficientdet.tools.create_labelmap:main',
              'prepare_voc = '
              'easy_efficientdet.tools.create_object_detection_dataset:main',
          ],
      })
