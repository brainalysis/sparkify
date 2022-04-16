from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

#with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
#    long_description = "\n" + fh.read()

def readme():
    with open("README.md", encoding="utf8") as f:
        README = f.read()
    return README

VERSION = "0.0.7"
DESCRIPTION = "For an easy implementation of spark's ml  "


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
