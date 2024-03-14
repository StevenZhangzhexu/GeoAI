import os
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point, Polygon
import alphashape as ash

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


def convert(bbox_dict, folder):
    name_dict = {
                    0: 'Bollard',
                    1: 'Building',
                    2: 'Bus Stop',
                    3: 'Control Box',
                    4: 'Ground',
                    5: 'Lamp Post',
                    6: 'Pole',
                    7: 'Railing',
                    8: 'Road',
                    9: 'Shrub',
                    10: 'Sign',
                    11: 'Solar Panel',
                    12: 'Tree'
                }
    keys = sorted(bbox_dict.keys())
    for label in tqdm(keys , desc= "Exporting Shapefile"):
        print(f'Processing {name_dict[label]} shape file')
        pgxyz_list =[]
        if label in (1,4,8):
            areas = [ bbox_dict[label][i][3] for i in range(len(bbox_dict[label])) ] 
            for area in areas:
                vector = ash.alphashape(area, alpha=0.5)
                pgxyz_list.append(vector)
            convert_to_shapefile_poly(pgxyz_list, folder, name_dict[label], crs = 'EPSG:3414')
        else:
            centroids = [ bbox_dict[label][i][2] for i in range(len(bbox_dict[label])) ]
            if not centroids:
                continue  # some pred points exist but could not form an object, skip
            convert_to_shapefile_cent(centroids, folder, name_dict[label], crs = 'EPSG:3414')



