# randlanet
conda create -n randlanet python=3.6
conda activate randlanet

pip install -r helper_requirements.txt
sh compile_op.sh
pip install tensorflow-gpu==1.*
pip install laspy[lazrs,laszip]

# check gpu with tf: python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# check gpu with tf: python -c "import tensorflow as tf; print(tf.__version__)"
# pip install open3d
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
