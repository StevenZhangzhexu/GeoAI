# randlanet
conda create -n orchard python=3.6
conda activate orchard

pip install -r helper_requirements.txt
sh compile_op.sh
pip install tensorflow==2.*
pip install laspy[lazrs,laszip]
