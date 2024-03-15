#!/bin/bash

conda create -n geoai -y python=3.6
conda init
conda activate geoai

pip install -r requirements.txt
sudo snap install ngrok

# randlanet
pip av_randlanet_scfnet/install -r requirements.txt
sh av_randlanet_scfnet/compile_op.sh
