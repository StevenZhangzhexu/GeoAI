#!/bin/bash

# randlanet installation
# manually install the below commented stuff first
# nvidia cuda driver and toolkit

conda create -n randlanet -y python=3.6
conda init
conda activate randlanet

pip install -r requirements.txt
sh compile_op.sh

# check gpu with tf: python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# check gpu with tf: python -c "import tensorflow as tf; print(tf.__version__)"
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
