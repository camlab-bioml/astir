import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="astir",
    version="0.0.1",
    author="Jinyu Hou",
    author_email="jhou@lunenfeld.ca",
    description=" ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/camlab-bioml/astir",
    packages=["astir"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="GPLv2",
    install_requires=[
        "torch",
        "pandas",
        "numpy",
        "nbformat",
        "pyyaml",
        "sklearn",
        "argparse",
        "matplotlib",
        "loompy",
        "tqdm",
        "anndata",
        "rootpath",
        "nbconvert",
        "h5py",
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite="nose.collector",
    test_require=["nose"],
    scripts=["bin/astir"],
)
