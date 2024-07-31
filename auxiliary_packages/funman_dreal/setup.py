import os
from pathlib import Path

from setuptools import find_packages, setup

with open(os.path.join("src", "funman_dreal", "_version.py")) as version_file:
    version = version_file.readlines()[-1].split()[-1].strip("\"'")

# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="funman_dreal",
    version=version,
    description="Functional Model Analysis Tool - DReal Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="Dan Bryce",
    author_email="dbryce@sift.net",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["funman", "docker", "tenacity", "pyparsing"],
    # extras_require={"dreal": ["dreal"]},
    tests_require=["unittest"],
    zip_safe=False,
)
