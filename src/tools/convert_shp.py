import os
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point,LineString, MultiPolygon
from shapely.ops import unary_union
import alphashape as ash
import math
import numpy as np
from scipy.spatial import cKDTree

def convert_to_shapefile_cent(xyz_list, output_folder, output_name, crs = 'EPSG:3414'): # default crs for svy21
    seperate_xyz = list(map(list, zip(*xyz_list)))
    geometry = [Point(xyz) for xyz in xyz_list]
    gdf = gpd.GeoDataFrame(
        {   'X': seperate_xyz[0], 
            'Y': seperate_xyz[1],
            'Z': seperate_xyz[2]
        },
        geometry=geometry,
        crs=crs
    )
   
    output_shapefile = os.path.join(output_folder, f"{output_name}.shp")
    output_folder = os.path.dirname(output_shapefile)
    os.makedirs(output_folder, exist_ok=True)

    gdf.to_file(output_shapefile)
    print('Done')


def convert_to_shapefile_poly(pgxyz_list, output_folder, output_name, crs = 'EPSG:3414'): # default crs for svy21
    # Create a GeoDataFrame
    geometry = []
    # Create a Shapely Polygon from the convex hull vertices
    for polygon in pgxyz_list:
        geometry.append(polygon)

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


def convert_to_shapefile_lines(pgxyz_list, output_folder, output_name, crs = 'EPSG:3414'): # default crs for svy21
    # Create a GeoDataFrame
    geometry = []
    # Create a Shapely Polygon from the convex hull vertices
    for polygon in pgxyz_list:
        exterior_coords = polygon.exterior.coords
        geometry.append(LineString(exterior_coords))

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

# def merge(polygons, centroids, distance_threshold=10):
#     merged = []
#     while len(polygons) > 0:
#         poly = polygons.pop()
#         cent = centroids.pop()
#         print('poly',poly)
#         print('bf,',len(polygons))
#         i = 0
#         while i < len(polygons):
#             print('dist',distance(cent,centroids[i]))
#             if distance(cent,centroids[i]) < distance_threshold:              
#                 poly = unary_union([poly, polygons.pop(i)])
#                 centroids.pop(i)
#                 print('aft,', len(polygons))
#             else:
#                 i += 1
#         merged.append(poly)
#         # print('res',len(merged))
#     return merged

# def distance(a,b):
#     """Calculate the Euclidean distance between two points."""
#     return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

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
        vector = ash.alphashape(area, alpha=0.1)
        if not isinstance(vector, MultiPolygon):  # exclude multipolygon
            pgxyz_list.append(vector)
    return pgxyz_list
    
def convert(bbox_dict, folder):
    name_dict = {
                    0: 'Bollard',
                    1: 'Building',
                    2: 'Bus Stop',
                    3: 'ControlBox',
                    4: 'Ground',
                    5: 'LampPost',
                    6: 'Pole',
                    7: 'Railing',
                    8: 'Road',
                    9: 'Shrub',
                    10: 'Sign',
                    11: 'SolarPanel',
                    12: 'Tree'
                }
    keys = sorted(bbox_dict.keys())
    for label in tqdm(keys , desc= "Exporting Shapefile"):
        print(f'Processing {name_dict[label]} shape file')
        if label in (1,2,4,8, 0,6):
            # areas = [ bbox_dict[label][i][3] for i in range(len(bbox_dict[label])) ] 
            # # for i, area in enumerate(areas):
            # for area in areas:
            #     vector = ash.alphashape(area, alpha=0.5)
            #     pgxyz_list.append(vector)

            area_dict = { tuple(bbox_dict[label][i][2]) :  bbox_dict[label][i][3] for i in range(len(bbox_dict[label])) }                 
            if label == 8:
                pgxyz_list = area_merge(area_dict, 10)
                convert_to_shapefile_lines(pgxyz_list, folder, name_dict[label], crs = 'EPSG:3414')
            elif label in (1,2,4):
                pgxyz_list = area_merge(area_dict, 10)
                convert_to_shapefile_poly(pgxyz_list, folder, name_dict[label], crs = 'EPSG:3414')
            else:      
                pgxyz_list = area_merge(area_dict, 0.1)          
                convert_to_shapefile_poly(pgxyz_list, folder, name_dict[label], crs = 'EPSG:3414')
        else:
            centroids = [ bbox_dict[label][i][2] for i in range(len(bbox_dict[label])) ]
            if not centroids:
                continue  # some pred points exist but could not form an object, skip
            convert_to_shapefile_cent(centroids, folder, name_dict[label], crs = 'EPSG:3414')
