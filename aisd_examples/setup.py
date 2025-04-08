from setuptools import setup

setup(
    name="aisd_examples",
    version="0.1",
    packages=["aisd_examples", "aisd_examples.envs"],
    install_requires=['gymnasium']
)