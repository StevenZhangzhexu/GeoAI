from UNext.vectorize import update_shp
import os

#chosen_folder ='UNext/results/test_input/SK_Test1_Pred_2024-06-05_07-46-13/'
chosen_folder ='UNext/results/test_input/SK_Test2_Pred_2024-06-05_05-02-07'

import os

# Get the absolute path of the current working directory
current_directory = os.path.abspath(".")
print(current_directory)

update_shp(output_folder = chosen_folder)