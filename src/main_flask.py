import os
from flask import *
from av_randlanet_scfnet import predict_OrchardRoad
from av_randlanet_scfnet.utils import data_prepare_orchard
from tools import shapefile_conversion, shapefile_conversion_ver2
from av_randlanet_scfnet.utils import helper_las, separate_predicted_objects
import subprocess
import threading
import time


app = Flask(__name__, static_folder='static/')


@app.route('/')
def login():
    return render_template("login.html")


@app.route('/home', methods=['POST'])
def upload():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        try:
            if (email.endswith("@alteredverse.net") or email.endswith("@yjpsurveyors.com")) and len(password) >= 8:
                return render_template("main.html")
            else:
                return render_template("error2.html", msg="Please contact the admin for further assistance.")
        except Exception as err:
            print(err)
            return render_template("error2.html", msg="Please provide correct credentials.")


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
            separate_predicted_objects.separate_and_segment_point_clouds(
                f.filename)

            # copy the results to shared folder
            helper_las.copy_predictions()

            # separate_predicted_objects.separate_and_cluster_point_cloud_objects(f.filename)
            # sam_instance_segmentation.run_sam_instance_segmentation(f.filename)
            subprocess.run(['/home/pc1/miniconda3/envs/samlidar/bin/python',
                           'av_randlanet_scfnet/utils/sam_instance_segmentation.py', f.filename])
            # subprocess.run(['conda', 'run', '-n', 'samlidar', 'av_randlanet_scfnet/utils/sam_instance_segmentation.py', f.filename])
            # separate_predicted_objects.separate_and_cluster_point_cloud_objects(f.filename)

            # shapefile conversion
            shapefile_conversion.convert_main(f.filename)
            shapefile_conversion_ver2.convert_main(f.filename)

            # copy the results to shared folder
            helper_las.copy_predictions()

            # visualize predictions
            # vis_pred_OrchardRoad.viz_pred(f.filename)
            # subprocess.run(['/home/pc1/miniconda3/envs/open3d/bin/python',
            #                 'tools/vis_pred_semseg_OrchardRoad.py', f.filename])
            # subprocess.run(['/home/pc1/miniconda3/envs/open3d/bin/python',
            #                 'tools/vis_pred_bboxes_OrchardRoad.py', f.filename])

            print("All finished!")

            # visualize prediction
            def thread_vis():
                print("Thread starting...")
                time.sleep(2)
                subprocess.run(['/home/pc1/miniconda3/envs/open3d/bin/python',
                            'tools/visualize_open3d_webrtc.py', f.filename])
                print("Thread finishing...")

            # close_port.close_port(8888)
            subprocess.run(['/home/pc1/miniconda3/envs/open3d/bin/python',
                            'tools/close_port.py'])
            time.sleep(2)
            x = threading.Thread(target=thread_vis)
            x.start()

            return render_template("success.html", name=f.filename)
        except Exception as err:
            print(err)
            return render_template("error.html", name=f.filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
