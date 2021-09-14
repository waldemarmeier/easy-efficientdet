# TODO major todo

# integrate scripts with entry point
# https://stackoverflow.com/questions/774824/explain-python-entry-points

from distutils.core import setup

setup(
    name='TowelStuff',
    version='0.1dev',
    packages=[
        'towelstuff',
    ],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.txt').read(),
)
