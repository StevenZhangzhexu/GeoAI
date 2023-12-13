# randlanet installation
# manually install the below commented stuff first
# nvidia cuda driver and toolkit

conda create -n samlidar -y python=3.10
conda activate samlidar

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install segment-lidar
pip install open3d
