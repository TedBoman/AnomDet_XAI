from setuptools import setup, find_packages

setup(
    name="mypackage",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "examples*", "scripts/*"]),
)