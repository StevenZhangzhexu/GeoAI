import os
from flask import *
from av_randlanet_scfnet import predict_OrchardRoad
from av_randlanet_scfnet.utils import data_prepare_orchard, separate_predicted_objects, helper_las


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
            upload_path = 'av_randlanet_scfnet/data/orchard_road/test_inputs'
            file_path = os.path.join(upload_path, f.filename)
            os.makedirs(upload_path, exist_ok=True)
            f.save(file_path)

            # pre-process
            data_prepare_orchard.prepare_data(file_path)

            # predict
            predict_OrchardRoad.predict(filepath=file_path)

            # post-process

            # separate_predicted_objects.separate_segmented_point_clouds(f.filename)
            # separate_predicted_objects.separate_and_cluster_point_clouds(f.filename)
            separate_predicted_objects.separate_and_segment_point_clouds(f.filename)

            # copy the results to shared folder
            helper_las.copy_predictions()

            separate_predicted_objects.separate_and_cluster_point_cloud_objects(f.filename)

            # copy the results to shared folder
            helper_las.copy_predictions()

            print("All finished!")

            return render_template("success.html", name=f.filename)
        except Exception as err:
            print(err)
            return render_template("error.html", name=f.filename)


if __name__ == '__main__':  
    app.run(host='0.0.0.0', port=8001, debug=True)
