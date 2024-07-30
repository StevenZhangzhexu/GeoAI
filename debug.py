import os
from flask import *
from UNext import main_Unext_infer

# filename = 'SK_ZoneA.laz'
folder_name = 'Bishan' #'test_input'
# os.makedirs(upload_path, exist_ok=True)
# files = ['test_input/SK_Test1.laz','test_input/SK_Test2.laz' ]
files = ['Bishan/ZoneA_20240220_BSH_000087.laz']
for filename in files:
    # filename = file.split('/')[-1]
    upload_path = f'UNext/test_inputs/{folder_name}'
    file_path = os.path.join('UNext/test_inputs/', filename) 

    # predict
    main_Unext_infer.predict(filepath=file_path,uploadpath=upload_path)

download_path = f'UNext/test_outputs/{folder_name}'
os.makedirs(download_path, exist_ok=True)
main_Unext_infer.shape_output(files, download_path)

# copy the results to shared folder
# helper_las.copy_predictions()

# post-process

# copy the results to shared folder
# helper_las.copy_predictions()








