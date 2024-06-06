# To test/finetune Clustering & shp files conversion after SemSeg
import os,glob
from os.path import join
import logging
from tqdm import tqdm
import numpy as np
import laspy
from UNext.utils.bbox_ import bbox_pcd
import pickle
import geopandas as gpd
from shapely.ops import nearest_points, linemerge, unary_union
from shapely.geometry import Point,LineString, Polygon
import alphashape as ash
import numpy as np
from scipy.spatial import cKDTree
from centerline.geometry import Centerline
from collections import defaultdict

log = logging.getLogger(__name__)

# -------------- utils ----------------

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
        if not polygon.is_empty:
            exterior_coords = polygon.exterior.coords
            geometry.append(LineString(exterior_coords))
        else:
            print(polygon)

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
        if isinstance(vector, Polygon):
            # print(type(vector))  
            pgxyz_list.append(vector)
    return pgxyz_list

# -------------- main function ----------------


def convert(bbox_dict, name_dict, folder, skip=set()):
    keys = bbox_dict.keys()
    keys = sorted(keys - skip)
    print('------------------------------------') #

    vaules_dict = {v:k for k,v in name_dict.items()}
    cust_values = ['Building', 'BusStop', 'Ground', 'Road', 'Bollard', 'Pole', 'Railing', 'Hydrant', 'Shed', 'Overpass','PedestrianOverheadBridge','ZebraBeaconPole', 'Barrier','CoveredLinkway','Pathway']
    cust_lb = [vaules_dict[v] for v in cust_values if v in vaules_dict]
    for label in tqdm(keys , desc= "Exporting Shapefile"):
        if label == 'Unclassified':
            continue
        elif label in cust_lb:
            print(f'Processing {name_dict[label]} shape file -- Geometry')
            area_dict = { tuple(bbox_dict[label][i][2]) :  bbox_dict[label][i][3] for i in range(len(bbox_dict[label])) }         
            if name_dict[label] == 'Road':
                pgxyz_list = area_merge(area_dict, 10)
                # print(pgxyz_list)
                convert_to_shapefile_lines(pgxyz_list, folder, name_dict[label], crs = 'EPSG:3414')
            elif name_dict[label] in ('Building', 'Bus Stop','Overpass', 'PedestrianOverheadBridge', 'Ground','Railing','CoveredLinkway','Pathway'):
                pgxyz_list = area_merge(area_dict, 10)
                convert_to_shapefile_poly(pgxyz_list, folder, name_dict[label], crs = 'EPSG:3414')
            else:      
                pgxyz_list = area_merge(area_dict, 0.1)          
                convert_to_shapefile_poly(pgxyz_list, folder, name_dict[label], crs = 'EPSG:3414')
        else:
            print(f'Processing {name_dict[label]} shape file. -- Points')
            centroids = [ bbox_dict[label][i][2] for i in range(len(bbox_dict[label])) ]
            if not centroids:
                continue  # some pred points exist but could not form an object, skip
            convert_to_shapefile_cent(centroids, folder, name_dict[label], crs = 'EPSG:3414')

    print('------------------------------------')

def bbox_to_shp(filename, name_dict, restore=False, output_folder = None, las_data = None):
    os.makedirs(output_folder, exist_ok=True)
    bbox_dict, skip = None, set()
    print("Creating Bounding Boxes...")
    if restore: # cached bbox 
        pkl_path = os.path.join(output_folder, 'bbox_dict.pkl')
        with open(pkl_path, 'rb') as file:
            bbox_dict = pickle.load(file)   
        skip = {11,14,15}

    # clustering
    if not las_data:
        path = os.path.join(output_folder,filename)
        if '.laz' not in path:
            path += '.laz'
        las_data = laspy.read(path)

    bbox_dict = bbox_pcd(las_data, name_dict, False, False, restore=restore, bbox=bbox_dict, skip=skip)
    file_path = os.path.join(output_folder, 'bbox_dict.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(bbox_dict, file)

    shp_folder = output_folder + '/shp' 
    convert(bbox_dict, name_dict, folder=shp_folder, skip=skip)


# -------------- post process ----------------

def update_shp(zone = None, output_folder = None):
    shp_folder = output_folder + '/shp'
    loc = False
    try :
        zone = shp_folder + '/Road.shp'
        zone_gdf = gpd.read_file(zone)
        loc = True
    except:
        print('No Road Found', zone)
        return

    ###########
    # Bus stop #
    ###########
    bsp = shp_folder + '/BusStop.shp'
    if os.path.exists(bsp):
        shapefile_gdf = gpd.read_file(bsp)
        lengths, widths, nearest_road_names = [],[],[]
        for index, row in shapefile_gdf.iterrows():
            # Get the geometry of the current line
            line_geometry = row['geometry']
            width = 2 * line_geometry.minimum_rotated_rectangle.exterior.distance(line_geometry.centroid)
            width = round(width, 7)
            length = (float(line_geometry.length) - 2*width)/2
            length = round(length, 7)
            if length < 3.8:
                shapefile_gdf.drop(index, inplace=True)
                continue
            widths.append(width)
            lengths.append(length)

        shapefile_gdf['Length'] = lengths
        shapefile_gdf['Width'] = widths
        # if loc:   
        #     shapefile_gdf['LocatName'] = nearest_road_names
        if shapefile_gdf.empty:
            print(f'{bsp} is empty')
        else:
            shapefile_gdf.to_file(bsp) # update bus stop shp
    else:
        print(f'no {bsp}')

    ###################
    # base edge point #
    ###################
    for shp_type in ['Bollard', 'ZebraBeaconPole', 'Pole']:
        shp_name = f'/{shp_type}.shp'
        shapefile_name = shp_folder + shp_name
        convert_polyg2edgepoint(shapefile_name, zone_gdf)
    
    ###################
    # centerline      #
    ###################
    for shp_type in ['Railing', 'Barrier','CoveredLinkway','Pathway']:
        shp_name = f'/{shp_type}.shp'
        shapefile_name = shp_folder + shp_name
        convert_polyg2ctl(shapefile_name)

    return

def convert_polyg2edgepoint(shapefile_name, zone_gdf):
    if os.path.exists(shapefile_name):
        shapefile_gdf = gpd.read_file(shapefile_name)
        if not shapefile_gdf.empty:
            updated_geometries = []
            for index, row in shapefile_gdf.iterrows():
                line_geometry = row['geometry']
                nearest_point = nearest_points(line_geometry, zone_gdf.unary_union)[0]
                updated_geometry = Point(nearest_point.coords[0])
                updated_geometries.append(updated_geometry)
                
            shapefile_gdf['geometry'] = updated_geometries
            shapefile_gdf.to_file(shapefile_name) 
        else:
            print(f'{shapefile_name} is empty')
    else:
        print(f'no {shapefile_name}')

def convert_polyg2ctl(shapefile_name):
    if os.path.exists(shapefile_name):
        shapefile_gdf = gpd.read_file(shapefile_name)
        if not shapefile_gdf.empty:
            updated_geometries = []
            for index, row in shapefile_gdf.iterrows():
                # Get the geometry of the current line
                line_geometry = row['geometry']
                try :
                    updated_geometry = Centerline(line_geometry)
                except:
                    shapefile_gdf.drop(index, inplace=True)
                    print('--------------- pass ctl')
                    continue

                centerline_line_strings = list(updated_geometry.geoms)
                # print(centerline_line_strings)
                threshold_length = 0.01  # Adjust this threshold as needed
                filtered_line_strings = [line for line in centerline_line_strings if line.length > threshold_length]
                merged_line = linemerge(filtered_line_strings)
                updated_geometries.append(merged_line)

            shapefile_gdf['geometry'] = updated_geometries
            shapefile_gdf.set_geometry(col='geometry', inplace=True)
            if shapefile_gdf.empty:
                print(f'{shapefile_name} is empty after processing')
            else:
                shapefile_gdf.to_file(shapefile_name)
        else:
            print(f'{shapefile_name} is empty')
    else:
        print(f'no {shapefile_name}')

def merge_shp(filepaths, name_dict, download_path):
    final_bbox = defaultdict(list)
    res_folders = []
    for filepath in filepaths:
        filename = os.path.splitext(filepath.split('/')[-1])[0]
        folder = filepath.split('/')[-2]
        subfolders = glob.glob(os.path.join('UNext/results', f"{folder}", f"{filename}_Pred_*"))
        res_folders.append(max(subfolders, key=os.path.getctime))
    
    for subfolder in res_folders:
        bbox_path = os.path.join(subfolder, 'bbox_dict.pkl')
        if os.path.exists(bbox_path):
            print('Reading', bbox_path) ##############3
            with open(bbox_path, 'rb') as f:
                bbox_dict = pickle.load(f)
            for key, value in bbox_dict.items():
                final_bbox[key].extend(value)
    # os.makedirs(download_path, exist_ok=True)
    op_path = os.path.join(download_path, 'bbox_dict.pkl')
    with open(op_path, 'wb') as file:
        pickle.dump(final_bbox, file)
    shp_folder = download_path + '/shp' 
    convert(final_bbox, name_dict, folder=shp_folder)
    update_shp(output_folder = download_path)



