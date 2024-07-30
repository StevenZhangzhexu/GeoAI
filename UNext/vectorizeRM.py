# To test/finetune Clustering & shp files conversion after SemSeg
import os
from os.path import join
import sys
import logging
from tqdm import tqdm
from sklearn.cluster import DBSCAN  # import open3d as o3d #just switch to open3d.ml.tf for tf usage
import numpy as np
import laspy
import pickle
import geopandas as gpd
from shapely.ops import nearest_points, linemerge, unary_union
from shapely.geometry import Point,LineString, Polygon
import alphashape as ash
from scipy.spatial import cKDTree
from centerline.geometry import Centerline
# from pygeoops import centerline as Centerline



log = logging.getLogger(__name__)

import numpy as np
import laspy
# from scipy.spatial import ConvexHull
# from scipy import stats
from collections import defaultdict


def bbox_pcd(pc_path, name_dict,visualize = False, visualize_by_cat = False):
    # Create a list to store all bounding box geometries, centroids, and point counts
    bounding_boxes = []
    centroids = []
    point_counts = []
    lbs=[]
    vol=[]
    cat_instances = defaultdict(list)

    las_data = laspy.read(pc_path)
    # all_points = np.vstack((las_data.x, las_data.y, las_data.z)).T
    labels = las_data.pred #label #
    print(set(labels))
    # max bbox vol
    all_points = np.column_stack((las_data.x, las_data.y, las_data.z))
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    max_dimensions = max_coords - min_coords
    
    epsilon = 1
    minpts = 10
    

    for tag in sorted(set(labels)):
        if name_dict[tag] == 'OthersRoad':
            continue
        class_indices = (labels == tag) 
        class_points = np.column_stack((las_data.x[class_indices], las_data.y[class_indices], las_data.z[class_indices]))  
        if len(class_points) == 0:
            continue     

        #############
        # clustering #
        #############

        # Downsampling (xyz)
        voxel_size = 0.01 #0.1
        voxel_grid = np.floor(class_points / voxel_size).astype(int)
        unique_voxels = np.unique(voxel_grid, axis=0)
        downsampled_points = unique_voxels * voxel_size + voxel_size / 2
        db = DBSCAN(eps=epsilon, min_samples=minpts).fit(downsampled_points)
        clusters = db.labels_
        unique_cluster = np.unique(clusters)

        # Create bounding boxes for each cluster
        for cluster in unique_cluster:
            if cluster == -1:
                continue
            cluster_indices = np.where(clusters == cluster)[0]
            cluster_points = downsampled_points[cluster_indices]

            # Compute bounding box
            min_coords = np.min(cluster_points, axis=0)
            max_coords = np.max(cluster_points, axis=0)
            centroid = np.mean(cluster_points, axis=0)

            # Get point count
            point_count = len(cluster_points)

            area = np.asarray(cluster_points)[:,0:2]
            cat_instances[tag].append((min_coords, max_coords, centroid, area))

            point_counts.append(point_count)
            lbs.append(tag)

    for i, centroid in enumerate(centroids):
        print(f"Box {i+1} - Point Count: {point_counts[i]}, Label: {lbs[i]}")

    print(cat_instances.keys())
         
    return cat_instances



# -------------- converting----------------

def convert_to_shapefile_ctlines(pgxyz_list, output_folder, output_name, crs = 'EPSG:3414'): # default crs for svy21
    # Create a GeoDataFrame
    geometry = []
    # Create a Shapely Polygon from the convex hull vertices
    for geom in pgxyz_list:
        if geom.is_empty:
            print('geom is empty')
            continue
        if isinstance(geom, LineString):
            geometry.append(geom)
            continue

        try:
            ctls = Centerline(geom)
        except:
            print(f'fail to get centerline for {geom}')
            continue
        centerline_line_strings = list(ctls.geoms)
        threshold_length = 0.1  # Adjust this threshold as needed
        filtered_line_strings = [line for line in centerline_line_strings if line.length > threshold_length]
        merged_line = linemerge(filtered_line_strings)
        geometry.append(merged_line)


    gdf = gpd.GeoDataFrame(
        geometry=geometry,
        crs=crs
    )

    if gdf.empty:
        print(f"{output_name} GeoDataFrame is empty")
        return
   
   # Create the output folder if it doesn't exist
    output_shapefile = os.path.join(output_folder, f"{output_name}.shp")
    output_folder = os.path.dirname(output_shapefile)
    os.makedirs(output_folder, exist_ok=True)

    # Save the GeoDataFrame to a Shapefile
    gdf.to_file(output_shapefile)
    print('Done')

def are_points_collinear(points):
    if len(points) < 3:
        return True
    # Calculate the area of the triangle formed by (points[0], points[1], points[i]) for each i
    # If all areas are zero, the points are collinear
    p0, p1 = points[0], points[1]
    for p2 in points[2:]:
        if np.linalg.norm(np.cross(p1 - p0, p2 - p0)) > 1e-10:
            return False
    return True

def area_merge(area_dict, radius = 0):
    pgxyz_list =[]
    q_points = [key for key in area_dict]
    merged = [False for _ in q_points]
    kd_tree =  cKDTree(np.array(q_points))
    for qi in range(len(q_points)):
        if merged[qi]:
            continue
        center = q_points[qi]
        indices = kd_tree.query_ball_point(center, radius)
        idx = [i for i in indices if not merged[i]]
        points_within_radius = [q_points[id] for id in idx]
        for id in idx:
            merged[id] = True      
        merged[qi] = True
        area = np.vstack([area_dict[p] for p in points_within_radius]) # merge areas
        if are_points_collinear(area):
            line = LineString(area)
            pgxyz_list.append(line)
            continue
        vector = ash.alphashape(area, alpha=0.1)
        if isinstance(vector, Polygon):
            # print(type(vector))  
            pgxyz_list.append(vector)
        else:
            print('Not ploygon -- skip')
    return pgxyz_list
    
def convert(bbox_dict, name_dict, folder):
    keys = sorted(bbox_dict.keys())
    vaules_dict = {v:k for k,v in name_dict.items()}
    for label in tqdm(keys , desc= "Exporting Shapefile"):
        # if label <17:
        #     continue
        # if label in (1,2,4,8, 0,6,7, 14,15,17):
        if label == 'Road':
            continue
        else:
            print(f'Processing {name_dict[label]} shape file -- Geometry')
            area_dict = { tuple(bbox_dict[label][i][2]) :  bbox_dict[label][i][3] for i in range(len(bbox_dict[label])) }         
            print(label)
            pgxyz_list = area_merge(area_dict, 1)
            convert_to_shapefile_ctlines(pgxyz_list, folder, name_dict[label], crs = 'EPSG:3414')
             

def RM_bbox_to_shp(filename, name_dict, restore=False, chosen_folder = 'shape_fld/new'): 
    # os.makedirs(chosen_folder, exist_ok=True)
    pc_name = filename + '.laz'
    print(filename)
    print("Creating Bounding Boxes...")
    path = os.path.join(chosen_folder, pc_name)
    print(' path check', path)
    if restore: # cache
        pkl_path = os.path.join(chosen_folder, 'RM_bbox_dict.pkl')
        with open(pkl_path, 'rb') as file:
            bbox_dict = pickle.load(file)
    else: 
        # clustering
        bbox_dict = bbox_pcd(path, name_dict, False, False)
        file_path = os.path.join(chosen_folder, 'RM_bbox_dict.pkl')
        with open(file_path, 'wb') as file:
            pickle.dump(bbox_dict, file)

    shp_folder = chosen_folder + '/shp' # 
    convert(bbox_dict, name_dict, folder=shp_folder)



