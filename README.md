# GeoAI API Server
REST API server for serving the GeoAI models in production

## Installation
```
conda create -n randlanet python=3.6
conda activate randlanet
pip install -r requirements.txt
cd src/av_randlanet_scfnet/
pip install -r requirements.txt
```

```
conda create -n samlidar python=3
conda activate samlidar

```

## Usage
Production Flask server: `waitress-serve --host 127.0.0.1 main_flask:app`
Port forwarding: `ssh -R 80:localhost:8080 serveo.net`

Production ngrok file server: `ngrok http "file:///home/pc1/shared"`


