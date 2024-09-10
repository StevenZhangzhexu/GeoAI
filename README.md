# GeoAI Flask Application - MVP

## Directory Structure
```
GeoAI_UNext/
   |__ .gitignore
   |__ main_flask_unext.py
   |__ README.md
   |__ requirements.txt
   |__ static/		# Directory of frontend
   |   |__ css/
   |   |__ js/
   |   |__ images/
   |__ templates/
   |   |__...
   |
   |__ UNext/		# Directory of backend for Deep Learning
       |__ compile_op.sh
       |__ main_Unext_infer.py
       |__ tester_infer.py
       |__ vis_pred.py
       |__ vectorizeRM.py
       |__ bbox.py
       |__ vectorize.py
       |__ test_outputs/
       |__ checkpoints/
       |   |__ snapshots1/	# checkpoints of trained model
       |       |__ snap-10501.meta
       |       |__ snap-10501.index
       |       |__ snap-10501.data-00000-of-00001
       |       |__ checkpoint
       |   |__ ...
       |
       |__ networks/
       |   |__ RandLANet_UNext_inf.py
       |__ test_inputs/
       |   |__ ...   	# User input data
       |__ results/
       |   |__ folderName/
       |       |__ fileName_Pred_TimeStamp/
       |           |__ log_test.txt
       |           |__ bbox_dict.pkl
       |           |__ fileName.laz
       |           |__ shp/
       |               |__ predObject.shp
       |               |__ predObject.dbf
       |               |__ predObject.cpg
       |               |__ predObject.prj
       |__ utils/
           |__ helper_las.py
           |__ project_images_onto_pcd.py
           |__ data_prepare.py
           |__ helper_ply.py
           |__ bbox_.py
           |__ project_images_onto_sphere.py
           |__ helper_tf_util.py
           |__ helper_tool.py
           |__ cpp_wrappers/		# C++ code
           |   |__ compile_wrappers.sh
           |   |__ cpp_utils/
           |       |__ nanoflann/
           |           |__ nanoflann.hpp
           |       |__ cloud/
           |           |__ cloud.h
           |           |__ cloud.cpp
           |   |__ cpp_subsampling/
           |       |__ setup.py
           |       |__ wrapper.cpp
           |       |__ grid_subsampling.cpython-36m-x86_64-linux-gnu.so
           |       |__ build/
           |           |__ temp.linux-x86_64-3.6/
           |               |__ wrapper.o
           |               |__ cpp_wrappers/
           |                   |__ cpp_utils/
           |                       |__ cloud/
           |                           |__ cloud.o
           |                   |__ grid_subsampling/
           |                       |__ grid_subsampling.o
           |           |__ grid_subsampling/
           |               |__ grid_subsampling.h
           |               |__ grid_subsampling.cpp
           |__ nearest_neighbors/
               |__ knn_.cxx
               |__ nanoflann.hpp
               |__ setup.py
               |__ test.py
               |__ knn.pyx
               |__ knn_.h
               |__ KDTreeTableAdaptor.h
               |__ knn.cpp
               |__ lib/
                   |__ python/
                       |__ nearest_neighbors.cpython-36m-x86_64-linux-gnu.so
                       |__ KNN_NanoFLANN-0.0.0-py3.6.egg-info
               |__ build/
                   |__ temp.linux-x86_64-3.6/
                       |__ knn_.o
                       |__ knn.o
                   |__ lib.linux-x86_64-3.6/


```

## Installation
```
conda create -n server -y python=3.6
conda activate server
pip install open3d --default-timeout=100 --ignore-installed numpy --ignore-installed decorator --ignore-installed certifi future --no-deps
pip install -r requirements.txt

cd UNext/
sh compile_op.sh
```

## Usage
```
python main_flask_unext.py
```

## Reference
https://github.com/zeng-ziyin/U-Next

https://github.com/QingyongHu/RandLA-Net
