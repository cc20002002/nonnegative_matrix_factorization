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
    entry_points={
        "console_scripts": [
            "nmf = nmf.scripts.cli:cli",
        ]
    },
    install_requires=[
        "numpy>=1.14.0",
        "scipy>=1.0.0",
        "scikit-learn>=0.19.1",
        "click==6.7",
        "pandas>=0.20.2",
        "matplotlib>=2.0.2",
        "Pillow==5.2.0",
    ],
    # extras_require={
    #     "dev": [
    #         "ipython>=6.2.1",
    #         "pytest>=3.1.3",
    #         "pytest-flake8>=0.8.1",
    #         "pytest-mock>=1.6.2",
    #         "pytest-cov>=2.5.1",
    #         "pytest-regtest>=0.15.1",
    #         "flake8-docstrings>=1.1.0",
    #         "flake8-quotes>=0.11.0",
    #         "flake8-comprehensions>=1.4.1"
    #     ]
    # },
    license="All Rights Reserved",
    zip_safe=False,
    keywords="nmf",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Operating System :: POSIX",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
