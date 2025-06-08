# setup.py
from setuptools import setup

setup(
    name="simpatico",
    version="0.1",
    packages=["simpatico"],
    entry_points={
        "console_scripts": [
            "simpatico = simpatico.cli:main",
        ],
    },
    include_package_data=True,
)
