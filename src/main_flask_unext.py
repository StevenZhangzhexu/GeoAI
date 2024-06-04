import os
from flask import *
from UNext import main_Unext_infer
import subprocess
from tools.convert_shp import convert
import pickle
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
                return render_template("main_.html")
            else:
                return render_template("error2.html", msg="Please contact the admin for further assistance.")
        except Exception as err:
            print(err)
            return render_template("error2.html", msg="Please provide correct credentials.")


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # f = request.files['file']
        files = request.files.getlist('files[]')
        print(files)
        try:
            start_time = time.time()
            folder_name = os.path.normpath(files[0].filename).split(os.sep)[0] #os.path.splitext(files[0].filename)[0]
            upload_path = f'UNext/test_inputs/{folder_name}'
            os.makedirs(upload_path, exist_ok=True)
            for f in files:
                file_path = os.path.join('UNext/test_inputs/', f.filename) # filename : 'folder/file.laz'
                f.save(file_path)
                main_Unext_infer.predict(filepath=file_path, uploadpath=upload_path)
            
            download_path = f'UNext/test_outputs/{folder_name}'
            os.makedirs(upload_path, exist_ok=True)
            main_Unext_infer.shape_output(file_path, download_path)
            end_time = time.time()
            execution_time = end_time - start_time
            minutes = int(execution_time // 60)
            seconds = int(execution_time % 60)
            print("All finished!")
            print(f"Total execution time of {folder_name}: {minutes} mins {seconds} s")


            # visualize prediction
            def thread_vis():
                print("Thread starting...")
                subprocess.run(['/home/steven/miniconda3/envs/vis/bin/python',
                            'tools/visualize_open3d_webrtc.py', f.filename])
                print("Thread finishing...")

            # close_port.close_port(8888)
            subprocess.run(['/home/steven/miniconda3/envs/vis/bin/python',
                            'tools/close_port.py'])
            x = threading.Thread(target=thread_vis)
            x.start()

            return render_template("success.html", name=f.filename)
        except Exception as err:
            print(err)
            return render_template("error.html", name=f.filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
