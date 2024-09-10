import os
from UNext import main_Unext_infertest as main_Unext_infer
import time
import glob


if __name__ == '__main__':

    start_time = time.time()
    # folder_name = os.path.normpath(files[0].filename).split(os.sep)[0] #os.path.splitext(files[0].filename)[0]
    # upload_path = os.path.join('UNext', 'test_inputs', folder_name)
    upload_path = os.path.join('UNext', 'test_inputs', 'bstest') #'Bishan')
    # os.makedirs(upload_path, exist_ok=True)
    
    # for f in files:
    #     file_path = os.path.join('UNext', 'test_inputs', f.filename) # filename : 'folder/file.laz'
    for file_path in glob.glob(os.path.join(upload_path, '*.laz')):
        print('Processing', file_path)
        main_Unext_infer.predict(filepath=file_path, uploadpath=upload_path)
    
    # combine whole area shape
    # download_path = os.path.join(executable_dir, 'UNext', 'test_outputs', folder_name)      
    # os.makedirs(download_path, exist_ok=True)
    # main_Unext_infer.shape_output(files, upload_path, download_path)
    # print('ckp1')
    # main_Unext_infer.RM_shape_output(files, download_path)
    # print('ckp2')

    end_time = time.time()
    execution_time = end_time - start_time
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    print("All finished!")
    print(f"Total execution time: {hours} hours {minutes} mins {seconds} s")

            
