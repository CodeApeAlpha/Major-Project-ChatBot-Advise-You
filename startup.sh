#!/bin/bash

# Command to update pip
python -m pip install --upgrade pip

pip pysqlite3-binary

# Command to install packages from a requirements file
pip install -r requirements.txt