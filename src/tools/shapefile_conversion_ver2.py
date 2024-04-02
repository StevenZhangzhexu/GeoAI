import os
import sys
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point, Polygon
import alphashape as ash
import open3d as o3d
import numpy as np
import laspy
import pickle
from collections import defaultdict


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


def create_bounding_box(min_bound, max_bound):
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    points = np.array([
        [min_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], max_bound[1], max_bound[2]],
        [min_bound[0], max_bound[1], max_bound[2]]
    ])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set


def get_kitti_bbox_info(file_path):
    # Load the .laz file
    cloud = laspy.read(file_path)
    points = np.vstack((cloud.x, cloud.y, cloud.z)).T

    # Calculate center point
    center = np.mean(points, axis=0)

    # Apply PCA and extract eigenvectors
    # Compute the covariance matrix
    cov = np.cov(points, rowvar=False)

    # Perform eigenvalue decomposition to get the rotation matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    rot_mat = eigenvectors

    # Transform points using the rotation matrix
    rotated_points = np.dot(points, rot_mat)

    # Calculate box size aligned with the elongation
    box_size = np.max(rotated_points, axis=0) - np.min(rotated_points, axis=0)

    print(center, rot_mat, box_size)
    bbox = o3d.geometry.OrientedBoundingBox(center, rot_mat, box_size)

    return bbox


def get_base_center_for_tree_lamppost(object_points):
    # Find the highest point (top)
    highest_point = object_points[np.argmax(object_points[:, 2])]

    # Estimate the base center by considering points at the lowest height
    lowest_height = np.percentile(object_points[:, 2], 5)  # Assuming 5% height from the lowest
    base_points = object_points[object_points[:, 2] < lowest_height]

    # Calculate the center of these base points
    center_xy = np.mean(base_points[:, :2], axis=0)  # Consider only X and Y coordinates

    print(f"Highest point: {highest_point}")
    print(f"Estimated base center: {center_xy}")

    return center_xy


def get_center_base_coords(coordinates, label_id):
    # Find the point with the lowest z-coordinate (assuming z represents height)
    base_point = coordinates[np.argmin(coordinates[:, 2])]
    labels_with_elongations = [5, 6, 10, 11, 12]

    # check if with elongations
    if label_id in labels_with_elongations:
        center_coordinate = get_base_center_for_tree_lamppost(coordinates)
    else:
        center_coordinate = np.mean(coordinates, axis=0)
    print(center_coordinate)

    center = {
        'x': center_coordinate[0],
        'y': center_coordinate[1],
        'z': base_point[2],
    }

    return center


def bbox_pcd(filename):
    # Create a list to store all bounding box geometries, centroids, and point counts
    bounding_boxes = []
    centroids = []
    point_counts = []
    lbs = []
    cat_instances = defaultdict(list)

    ############
    # read file #
    ############
    directory_path = f'av_randlanet_scfnet/results/{filename}/separate_objects/'
    main_pcd_path = f'av_randlanet_scfnet/results/{filename}/predictions/{filename}'

    main = laspy.read(main_pcd_path)
    coords = np.vstack((main.x, main.y, main.z)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)

    for tag in name_dict.keys():
        print('tag', tag, 'class', name_dict[tag])
        for fname in os.listdir(directory_path):
            if tag in fname and not fname.endswith("_WGS84.laz"):
                try:
                    laz_file_path = os.path.join(directory_path, fname)
                    bbox = get_kitti_bbox_info(laz_file_path)

                    # read laz
                    data = laspy.read(laz_file_path)
                    points = np.vstack((data.x, data.y, data.z)).T

                    # Compute bounding box
                    min_coords = np.min(points, axis=0)
                    max_coords = np.max(points, axis=0)
                    max_dimensions = max_coords - min_coords

                    # Get point count
                    point_count = len(points)
                    centroid = get_center_base_coords(points, tag)

                    # Add bounding box, centroid, and point count to the respective lists
                    # Manually remove large box for some objects
                    dimensions = max_coords - min_coords
                    # volume = dimensions[0] * dimensions[1] * dimensions[2]
                    if tag not in (4, 8) and np.any(dimensions > 0.6 * max_dimensions):
                        continue
                    # Manually remove small box for some objects
                    elif tag in (1, 4, 8):
                        area = np.asarray(points)[:, 0:2]
                        cat_instances[tag].append((min_coords, max_coords, centroid, area))
                    else:
                        cat_instances[tag].append((min_coords, max_coords, centroid))

                    bounding_boxes.append(bbox)
                    centroids.append(centroid)
                    point_counts.append(point_count)
                    lbs.append(tag)
                except Exception as err:
                    print(err)

    # Print centroids and point counts
    for i, centroid in enumerate(centroids):
        print(f"Box {i + 1} - Centroid: {centroid}, Point Count: {point_counts[i]}. Label: {lbs[i]}")

    print(cat_instances.keys())

    return cat_instances


def compute_centroid(points):
    # Select any three points on the arc
    point1 = points[0]
    point2 = points[len(points) // 2]
    point3 = points[-1]

    # Construct two vectors
    vector1 = point2 - point1
    vector2 = point3 - point1

    # Compute normal vector to the plane defined by these points
    normal = np.cross(vector1, vector2)

    # Compute centroid (midpoint along the normal vector)
    centroid = point1 + 0.5 * normal  # Midpoint formula

    return centroid


def convert_to_shapefile_cent(xyz_list, output_folder, output_name, crs='EPSG:3414'):  # default crs for svy21
    seperate_xyz = list(map(list, zip(*xyz_list)))
    geometry = [Point(xyz) for xyz in xyz_list]
    gdf = gpd.GeoDataFrame(
        {
            'X': seperate_xyz[0],
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


def convert_to_shapefile_poly(pgxyz_list, output_folder, output_name, crs='EPSG:3414'):  # default crs for svy21
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


def convert_main(filename):
    pred_path = os.path.join("av_randlanet_scfnet/results/", filename, "predictions", filename)
    bbox_dict = bbox_pcd(pred_path)
    out_folder = os.path.join("av_randlanet_scfnet/results/", filename, "shapefiles_v2")
    os.makedirs(out_folder, exist_ok=True)
    file_path = os.path.join(out_folder, 'bbox_dict.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(bbox_dict, file)

    keys = sorted(bbox_dict.keys())
    for label in tqdm(keys, desc="Exporting Shapefile"):
        print(f'Processing {name_dict[label]} shape file')
        pgxyz_list =[]
        if label in (1, 4, 8):
            areas = [bbox_dict[label][i][3] for i in range(len(bbox_dict[label]))]
            for area in areas:
                vector = ash.alphashape(area, alpha=0.5)
                pgxyz_list.append(vector)
            convert_to_shapefile_poly(pgxyz_list, out_folder, name_dict[label], crs='EPSG:3414')
        else:
            centroids = [bbox_dict[label][i][2] for i in range(len(bbox_dict[label]))]
            if not centroids:
                continue  # some pred points exist but could not form an object, skip
            convert_to_shapefile_cent(centroids, out_folder, name_dict[label], crs='EPSG:3414')


if __name__ == '__main__':
    # convert_main("Orchard_0913_labelled_E.laz")
    try:
        convert_main(sys.argv[1])
    except Exception as err:
        print(err)
