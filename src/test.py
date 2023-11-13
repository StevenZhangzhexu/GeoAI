import time
from av_randlanet_scfnet import predict_OrchardRoad
from av_randlanet_scfnet.utils import data_prepare_orchard, separate_predicted_objects


start = time.time()
print("Starting...")
filename = 'Orchard_0913_labelled_C.laz'
file_path = 'av_randlanet_scfnet/data/orchard_road/test_inputs/Orchard_0913_labelled_C.laz'

# pre-process
print("pre-processing data...")
data_prepare_orchard.prepare_data(file_path)

# predict
print("predicting segmentation...")
predict_OrchardRoad.predict(filepath=file_path)

# post-process
print("post-processing results...")
separate_predicted_objects.separate_segmented_point_clouds(filename)

print("Finished!")
print("Time taken:", time.time() - start)
