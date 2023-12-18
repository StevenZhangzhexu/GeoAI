# GeoAI API Server
REST API server for serving the GeoAI models in production

## Installation
```
conda create -n randlanet -y python=3.6
conda activate randlanet
pip install -r requirements.txt
pip install -r src/av_randlanet_scfnet/requirements_main.txt
```

```
conda create -n samlidar -y python=3.10
conda activate samlidar
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install segment-lidar
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

```
conda create -n open3d -y python=3.10
conda activate open3d
pip install torch==1.13.* torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install open3d tensorboard laspy[lazrs,laszip]
```

## Usage
Production Flask server: `waitress-serve --host 127.0.0.1 main_flask:app`
Port forwarding: `ssh -R 80:localhost:8080 serveo.net`

Production ngrok file server: `ngrok http "file:///home/pc1/shared"`

