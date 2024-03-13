import os
from flask import *
from av_randlanet_scfnet import predict_OrchardRoad
from UNext import main_Unext_infer
from av_randlanet_scfnet.utils import data_prepare_orchard, separate_predicted_objects, helper_las
import subprocess

app = Flask(__name__, static_folder='static/')

@app.route('/')
def upload():
    return render_template("file_upload_form1.html")

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        f = request.files['file']
        try:
            # Upload file
            upload_path = 'UNext/test_inputs'
            file_path = os.path.join(upload_path, f.filename)
            os.makedirs(upload_path, exist_ok=True)
            f.save(file_path)

            # Pre-process
            data_prepare_orchard.prepare_data(file_path)

            # Determine which model to use
            option = request.form.get('option')  
            print('option,', option)

            # Predict
            if option == 'model0':
                predict_OrchardRoad.predict(filepath=file_path)
            elif option == 'model1':
                main_Unext_infer.predict(filepath=file_path, id=1)
            elif option == 'model2':
                main_Unext_infer.predict(filepath=file_path, id=2)
            elif option == 'model3':
                main_Unext_infer.predict(filepath=file_path, id=3)
            elif option == 'model4':
                main_Unext_infer.predict(filepath=file_path, id=4)

            # Post-process
            # separate_predicted_objects.separate_and_segment_point_clouds(f.filename)
            # helper_las.copy_predictions()

            # Run other scripts using subprocess if needed
            # subprocess.run(['/home/steven/miniconda3/envs/samlidar/bin/python', 
            #                 'av_randlanet_scfnet/utils/sam_instance_segmentation.py', f.filename])
            if option != 'model0':
                subprocess.run(['/home/steven/miniconda3/envs/vis/bin/python', 
                                'UNext/vis_pred.py', f.filename])
            # subprocess.run(['/home/steven/miniconda3/envs/vis/bin/python', 
            #                 'av_randlanet_scfnet/vis_pred_bboxes_OrchardRoad.py', f.filename])

            print("All finished!")
            return render_template("success.html", name=f.filename)
        except Exception as err:
            print(err)
            return render_template("error.html", name=f.filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
