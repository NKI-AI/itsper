#!/usr/bin/env python
# coding=utf-8
"""The setup script."""
import ast

from setuptools import setup

with open("itsper/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.s  # type: ignore
            break


with open("README.rst") as readme_file:
    long_description = readme_file.read()

install_requires = [
    "numpy>=1.25.2",
    "pillow>=9.5.0",
    "dlup>=0.3.34",
    "shapely",
    "pathlib",
    "typing",
]


setup(
    author="Ajey Pai Karkala",
    long_description=long_description,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        "console_scripts": [
            "itsper=itsper.cli:main",
        ],
    },
    description="An easy command line interface to compute the ITSP biomarker",
    install_requires=install_requires,
    license="MIT License",
    include_package_data=True,
    name="itsper",
    test_suite="tests",
    url="https://github.com/NKI-AI/itsper",
    py_modules=["itsper"],
    version=version,
)
