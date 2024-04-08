import os
from flask import *
from UNext import main_Unext_infer
from av_randlanet_scfnet.utils import data_prepare_orchard, helper_las
import subprocess
from tools.convert_shp import convert
import pickle
import threading


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
            # Upload file
            upload_path = f'UNext/test_inputs/{f.filename[:-4]}'
            file_path = os.path.join(upload_path, f.filename)
            os.makedirs(upload_path, exist_ok=True)
            f.save(file_path)

            # Pre-process
            data_prepare_orchard.prepare_data(pc_path=file_path, dataset_path=upload_path)

            # predict
            chosen_folder = main_Unext_infer.predict(filepath=file_path,uploadpath=upload_path, id=3)

            # copy the results to shared folder
            helper_las.copy_predictions_unext()

            # post-process
            subprocess.run(['/home/steven/miniconda3/envs/open3d/bin/python',
                            'UNext/vis_pred.py', f.filename])

            # shapefile conversion
            pkl_path = os.path.join(chosen_folder, 'bbox_dict.pkl')
            with open(pkl_path, 'rb') as file:
                bbox_dict = pickle.load(file)
            shp_folder = chosen_folder + '/shp'
            convert(bbox_dict, folder=shp_folder)

            # copy the results to shared folder
            helper_las.copy_predictions_unext()

            print("All finished!")

            # visualize prediction
            def thread_vis():
                print("Thread starting...")
                subprocess.run(['/home/pc1/miniconda3/envs/open3d/bin/python',
                            'tools/visualize_open3d_webrtc.py', f.filename])
                print("Thread finishing...")

            # close_port.close_port(8888)
            subprocess.run(['/home/pc1/miniconda3/envs/open3d/bin/python',
                            'tools/close_port.py'])
            x = threading.Thread(target=thread_vis)
            x.start()

            return render_template("success.html", name=f.filename)
        except Exception as err:
            print(err)
            return render_template("error.html", name=f.filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
