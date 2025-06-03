# setup.py
from setuptools import setup
from simpatico import config

setup(
    name="simpatico",
    version="0.1",
    packages=["simpatico"],
    entry_points={
        "console_scripts": [
            "simpatico = simpatico.cli:main",
        ],
    },
)
