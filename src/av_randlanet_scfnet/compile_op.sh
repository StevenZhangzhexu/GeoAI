#!/bin/bash

cd utils/nearest_neighbors || exit
python setup.py install --home="."
cd ../../

cd utils/cpp_wrappers || exit
sh compile_wrappers.sh
cd ../../../
