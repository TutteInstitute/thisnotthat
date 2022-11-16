from setuptools import setup, find_packages


setup(
    name="thisnotthat",
    packages=find_packages(),
    version="0.2",
    install_requires=[
        "bokeh",
        "panel",
        "param",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "numpy>=1.22",
        "umap-learn",
        "hdbscan",
        "colorcet",
        "cmocean",
        "glasbey",
    ],
)
