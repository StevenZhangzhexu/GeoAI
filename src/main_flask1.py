import os
from flask import *
from av_randlanet_scfnet import predict_OrchardRoad
from UNext import main_Unext_infer
from av_randlanet_scfnet.utils import data_prepare_orchard, separate_predicted_objects, helper_las
import subprocess
from convert_shp import convert
import pickle


app = Flask(__name__, static_folder='static/')


@app.route('/')
def upload():
    return render_template("file_upload_form1.html")


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        f = request.files['file']
        try:
            # Determine which model to use
            option = request.form.get('option')  
            print('option,', option)

            # Predict
            if option == 'model0':
                # upload file
                upload_path = 'av_randlanet_scfnet/data/orchard_road/test_inputs'
                file_path = os.path.join(upload_path, f.filename)
                os.makedirs(upload_path, exist_ok=True)
                f.save(file_path)
                # pre-process
                data_prepare_orchard.prepare_data(file_path)
                predict_OrchardRoad.predict(filepath=file_path)
            else:
                # Upload file
                upload_path = f'UNext/test_inputs/{f.filename[:-4]}'
                file_path = os.path.join(upload_path, f.filename)
                os.makedirs(upload_path, exist_ok=True)
                f.save(file_path)
                # Pre-process
                data_prepare_orchard.prepare_data(pc_path = file_path,  dataset_path= upload_path)

                if option == 'model1':
                    chosen_folder = main_Unext_infer.predict(filepath=file_path, uploadpath = upload_path, id=1)
                elif option == 'model2':
                    chosen_folder = main_Unext_infer.predict(filepath=file_path,uploadpath = upload_path, id=2)
                elif option == 'model3':
                    chosen_folder = main_Unext_infer.predict(filepath=file_path,uploadpath = upload_path, id=3)
                elif option == 'model4':
                    chosen_folder = main_Unext_infer.predict(filepath=file_path,uploadpath = upload_path, id=4)

            # Post-process
            # separate_predicted_objects 
            if option != 'model0':
                subprocess.run(['/home/steven/miniconda3/envs/vis/bin/python', 
                                'UNext/vis_pred.py', f.filename])                       
            # Convert objects to shp file
            if 'convertButton' in request.form:
                pkl_path = os.path.join(chosen_folder, 'bbox_dict.pkl')
                with open(pkl_path, 'rb') as file:
                    bbox_dict = pickle.load(file)
                shp_folder = chosen_folder +'/shp'  
                convert(bbox_dict, folder=shp_folder)
                
            print("All finished!")
            return render_template("success.html", name=f.filename)
        except Exception as err:
            print(err)
            return render_template("error.html", name=f.filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
