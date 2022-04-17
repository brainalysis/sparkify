from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))


def readme():
    with open("README.md", encoding="utf8") as f:
        README = f.read()
    return README


VERSION = "0.0.10"
DESCRIPTION = "For an easy implementation of spark's machine learning library"


# Setting up
setup(
    name="SparkAutoML",
    version=VERSION,
    author="Fahad Akbar",
    author_email="<fahadakbar@gmail.com>",
    description=DESCRIPTION,
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["pandas", "numpy", "pyspark >= 3.2.1", "pytest"],
    keywords=["python", "spark", "machine learning",],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
