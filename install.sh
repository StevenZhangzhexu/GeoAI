#!/bin/bash

# conda create -n geoai -y python=3.6
conda activate geoai

pip install -r requirements.txt
sudo snap install ngrok

# randlanet
pip install -r src/av_randlanet_scfnet/requirements_main.txt
sh src/av_randlanet_scfnet/compile_op.sh

$SHELL
