#!/usr/bin/env python
"""Setup environment for nmf."""

from setuptools import setup, find_packages

exec(open("nmf/__version__.py").read())
readme = open("README.md").read()

setup(
    name="nmf",
    version=__version__,
    description="Non-negative matrix factorization",
    long_description=readme,
    author="Joyce Wang",
    author_email="joyce.xinyue.wang@gmail.com",
    url="https://github.com/JoyceXinyueWang/nmf",
    packages=find_packages(),
    package_dir={"nmf": "nmf"},
    include_package_data=True,
    install_requires=[
        "numpy>=1.14.0",
        "scipy>=1.0.0",
        "scikit-learn>=0.19.1",
        "pandas>=0.20.2",
        "matplotlib>=2.0.2",
        "Pillow>=5.2.0",
        "tqdm>=4.26.0",
    ],
    license="All Rights Reserved",
    zip_safe=False,
    keywords="nmf",
)
