# setup.py
from setuptools import setup, find_packages

setup(
    name="fiztorch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
    ],
    author="ahammadnafiz",
    description="A toy implementation of PyTorch for educational purposes",
    python_requires=">=3.7",
)