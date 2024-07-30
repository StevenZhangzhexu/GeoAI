# GeoAI Flask Application - Dev Env

## Directory Structure
```
GeoAI_UNext/
|__ .gitignore # Ignored files and directories for Git
|__ main_flask_unext.py # Main Flask application script
|__ README.md # Project overview and documentation
|__ requirements.txt # Python package dependencies
|__ static/ # Directory of frontend
| |__ css/ # CSS files for styling
| |__ js/ # JavaScript files for interactivity
| |__ images/ # Image assets
|__ templates/ # HTML templates for rendering pages
| |__ ... # Other HTML templates
|
|__ UNext/ # Directory of backend for Deep Learning
|__ compile_op.sh # Script for compiling operations
|__ main_Unext_infer.py # Main script for UNext inference
|__ tester_infer.py # Testing script for UNext inference
|__ vis_pred.py # Visualization of predictions
|__ vectorizeRM.py # Script for vectorization
|__ bbox.py # Bounding box operations
|__ vectorize.py # Additional vectorization script
|__ test_outputs/ # Directory for output files from tests
|__ checkpoints/ # Checkpoints of trained model
| |__ snapshots1/ # Snapshots of model checkpoints
| |__ snap-10501.meta
| |__ snap-10501.index
| |__ snap-10501.data-00000-of-00001
| |__ checkpoint
| |__ ... # Additional checkpoints
|
|__ networks/ # Network definitions
| |__ RandLANet_UNext_inf.py
|__ test_inputs/ # User input data
| |__ ... # Input files for testing
|__ results/ # Results directory
| |__ folderName/ # Results for specific test runs
| |__ fileName_Pred_TimeStamp/
| |__ log_test.txt # Log file for test
| |__ bbox_dict.pkl # Pickled bounding box dictionary
| |__ fileName.laz # LAZ file for test
| |__ shp/ # Shapefile outputs
| |__ predObject.shp
| |__ predObject.dbf
| |__ predObject.cpg
| |__ predObject.prj
|__ utils/ # Utility scripts
|__ helper_las.py # Helper functions for LAS files
|__ project_images_onto_pcd.py # Project images onto point cloud data
|__ data_prepare.py # Data preparation script
|__ helper_ply.py # Helper functions for PLY files
|__ bbox_.py # Additional bounding box functions
|__ project_images_onto_sphere.py # Project images onto a sphere
|__ helper_tf_util.py # TensorFlow utility functions
|__ helper_tool.py # Miscellaneous helper functions
|__ cpp_wrappers/ # C++ code
| |__ compile_wrappers.sh
| |__ cpp_utils/ # C++ utility files
| |__ nanoflann/
| |__ nanoflann.hpp
| |__ cloud/
| |__ cloud.h
| |__ cloud.cpp
| |__ cpp_subsampling/ # C++ subsampling code
| |__ setup.py
| |__ wrapper.cpp
| |__ grid_subsampling.cpython-36m-x86_64-linux-gnu.so
| |__ build/ # Build directory for C++ code
| |__ temp.linux-x86_64-3.6/
| |__ wrapper.o
| |__ cpp_wrappers/
| |__ cpp_utils/
| |__ cloud/
| |__ cloud.o
| |__ grid_subsampling/
| |__ grid_subsampling.o
| |__ grid_subsampling/
| |__ grid_subsampling.h
| |__ grid_subsampling.cpp
|__ nearest_neighbors/ # Nearest neighbors C++ code
|__ knn_.cxx
|__ nanoflann.hpp
|__ setup.py
|__ test.py
|__ knn.pyx
|__ knn_.h
|__ KDTreeTableAdaptor.h
|__ knn.cpp
|__ lib/ # Library files
|__ python/
|__ nearest_neighbors.cpython-36m-x86_64-linux-gnu.so
|__ KNN_NanoFLANN-0.0.0-py3.6.egg-info
|__ build/ # Build directory for nearest neighbors code
|__ temp.linux-x86_64-3.6/
|__ knn_.o
|__ knn.o
|__ lib.linux-x86_64-3.6/
```

## Installation (U-Next & Conversion)
```
conda create -n server -y python=3.6
conda activate server
pip install open3d --default-timeout=100 --ignore-installed numpy --ignore-installed decorator --ignore-installed certifi future --no-deps
pip install -r requirements.txt

cd UNext/
sh compile_op.sh
```

## Usage of updated GeoAI Server (U-Next & Conversion)
```
python main_flask_unext.py
```