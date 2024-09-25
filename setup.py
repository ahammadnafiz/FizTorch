# setup.py
from setuptools import setup, find_packages

setup(
    name="FizTorch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",  # Add NumPy as a dependency
    ],
    author="Ahammad Nafiz",
    author_email="ahammadnafiz@outlook.com",
    description="A simple tensor library implementation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ahammadnafiz/FizTorch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)