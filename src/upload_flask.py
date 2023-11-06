import os
from flask import *
from av_randlanet_scfnet import predict_OrchardRoad
app = Flask(__name__)  


@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  


@app.route('/success', methods=['POST'])
def success():  
    if request.method == 'POST':  
        # upload file
        f = request.files['file']
        save_path = 'av_randlanet_scfnet/data/orchard_road/test_inputs'
        file_path = os.path.join(save_path, f.filename)
        os.makedirs(save_path, exist_ok=True)
        f.save(file_path)

        # predict and post-process
        predict_OrchardRoad.predict(filepath=file_path)

        # upload the predictions

        return render_template("success.html", name=f.filename)


if __name__ == '__main__':  
    app.run(host='0.0.0.0', port=8001, debug=True)
