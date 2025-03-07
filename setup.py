from setuptools import setup, find_packages

setup(
    name="DTractor",  # The name of your package
    version="0.1.0",
    description='DTractor package 0.1.0',
    author="Jin Kweon",
    author_email="yong.kweon@mail.mcgill.ca", 
    url="https://github.com/mcgilldinglab/DTractor",
    packages=find_packages(),  # Automatically finds the package (DTractor)
    classifiers=['Programming Language :: Python :: 3.9'],
    install_requires=[         # List any dependencies your package has
        "matplotlib==3.7.2",
        "numpy==1.23.5",
        "pandas==1.4.2",
        "plotly==5.6.0",
        "scanpy==1.9.5",
        "scikit_learn==1.0.2",
        "scipy==1.9.0",
        "scvi-tools == 0.20.2",
        "seaborn==0.11.2",
        "statsmodels==0.13.2",
        "torch==1.13.0",
        "jax==0.4.14",
        "jaxlib==0.4.14",
        "requests==2.32.3"
    ],
)

