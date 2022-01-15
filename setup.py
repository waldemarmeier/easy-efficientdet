from setuptools import find_packages, setup

requirements = {
    "default": ["tensorflow>=2.3.0", "matplotlib>=2.2.5"],
    "extra": {
        "eval": ["pycocotools>=2.0.2"]
    }
}


def load_readme() -> str:
    with open('README.md') as fp:
        return fp.read()


setup(
    name='easy-efficientdet',
    version='0.1',
    author="Waldemar Meier",
    author_email="info@waldemarmeier.com",
    url="https://github.com/waldemarmeier/easy-efficientdet",
    keywords="object-detection neural-network efficientdet",
    packages=find_packages(),
    license='Apache 2.0',
    python_requires='>=3.7',
    install_requires=requirements["default"],
    extras_require=requirements["extra"],
    description="Easy to use object detection package based on tensorflow",
    long_description=load_readme(),
    entry_points={
        'console_scripts': [
            'generate_tfrecord = easy_efficientdet.tools.generate_tfrecord:main',
            'create_labelmap = easy_efficientdet.tools.create_labelmap:main',
            'prepare_voc = '
            'easy_efficientdet.tools.prepare_voc:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
