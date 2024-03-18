#!/bin/bash

# randlanet installation
# manually install the below commented stuff first
# nvidia cuda driver and toolkit

conda create -n samlidar -y python=3.10
conda activate samlidar

# pip install torch==1.13.* torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install segment-lidar open3d tensorboard
