# randlanet installation
# manually install the below commented stuff first
# nvidia cuda driver and toolkit

conda create -n samlidar -y python=3.11
conda activate samlidar

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install open3d
pip install segment-lidar
