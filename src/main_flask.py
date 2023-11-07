import os
from flask import *
from av_randlanet_scfnet import predict_OrchardRoad
app = Flask(__name__, static_folder='static/')


@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        f = request.files['file']

        try:
            # upload file
            save_path = 'av_randlanet_scfnet/data/orchard_road/test_inputs'
            file_path = os.path.join(save_path, f.filename)
            os.makedirs(save_path, exist_ok=True)
            f.save(file_path)

            # predict and post-process
            predict_OrchardRoad.predict(filepath=file_path)

            # upload the predictions

            return render_template("success.html", name=f.filename)
        except Exception as err:
            print(err)
            return render_template("error.html", name=f.filename)


if __name__ == '__main__':  
    app.run(host='0.0.0.0', port=8001, debug=True)
