import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="py-shm-damage",
    version="1.0.6",
    description="Damage detection algorithms for damage detection in Structural Health Monitoring",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/carloshernangarrido/py-shm-damage",
    author="Hernán Garrido",
    author_email="carloshernangarrido@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["shmdamage"],
    include_package_data=True,
    install_requires=["scipy>=1.4.0"],
)
