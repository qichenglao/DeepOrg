from os import path
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open('requirements.txt') as f:
    requirements = f.read().split()

with open(path.join(here, 'VERSION')) as f:
    version = f.read().strip()

setup(
    name='DeepOrg',
    version=version,
    url='',
    license='',
    author='Qicheng Lao',
    author_email='qicheng.lao@gmail.com',
    description='Deep learning project organizer',
    packages=find_packages(
        exclude=['tests'],
    ),
    install_requires=requirements,
    include_package_data=True,
)
