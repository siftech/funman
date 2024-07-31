import os
from pathlib import Path

from setuptools import find_packages, setup

with open(os.path.join("src", "funman", "_version.py")) as version_file:
    version = version_file.readlines()[-1].split()[-1].strip("\"'")

# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="funman",
    version=version,
    description="Functional Model Analysis Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="Dan Bryce",
    author_email="dbryce@sift.net",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "graphviz",
        "multiprocess",
        "tenacity",
        "pyparsing",
        "pysmt @ git+https://github.com/danbryce/pysmt@add-abs-value#egg=pysmt-0.9.6.dev1",
        "pandas",
        "matplotlib",
        "pydantic>=2",
        "sympy",
        # "automates @ https://github.com/danbryce/automates/archive/e5fb635757aa57007615a75371f55dd4a24851e0.zip#sha1=f9b3c8a7d7fa28864952ccdd3293d02894614e3f"
    ],
    # dependency_links=["git+https://github.com/danbryce/pysmt/pysmt@add-abs-value#egg=pysmt-0.9.6.dev1"],
    extras_require={"dreal": ["funman_dreal"]},
    tests_require=["unittest", "pytest-cov", "scipy"],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "funman=funman.api.run:main",
        ],
    },
)
