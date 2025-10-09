#!/usr/bin/env bash

exit on error
set -o errexit

Install system dependencies required by scipy and numpy
apt-get update && apt-get install -y gfortran libopenblas-dev liblapack-dev

Install Python dependencies
pip install -r requirements.txt
