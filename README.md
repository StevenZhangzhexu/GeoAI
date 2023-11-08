# GeoAI API Server
REST API server for serving the GeoAI models in production

## Installation
`pip install -r requirements.txt`

## Usage
Production Flask server: `waitress-serve --host 127.0.0.1 main_flask:app`
Port forwarding: `ssh -R 80:localhost:8080 serveo.net`

Production ngrok file server: `ngrok http "file:///home/pc1/shared"`

