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
import traceback
from collections import defaultdict


name_dict = {
        0: 'Bollard',
        1: 'Building',
        2: 'BusStop',
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

label_to_min_points = {
                    0: 1000,
                    1: 5000,
                    2: 5000,
                    3: 3000,
                    4: 4000,
                    5: 3000,
                    6: 2000,
                    7: 5000,
                    8: 5000,
                    9: 3000,
                    10: 3000,
                    11: 4000,
                    12: 10000
                }
labels_with_elongations = [5, 6, 10, 11, 12]
labels_with_orientations = [5, 6, 10]


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

    # check if with elongations
    if label_id in labels_with_elongations:
        center_coordinate = get_base_center_for_tree_lamppost(coordinates)
    else:
        center_coordinate = np.mean(coordinates, axis=0)
    print(center_coordinate)

    base_center = [center_coordinate[0], center_coordinate[1], base_point[2]]

    return np.asarray(base_center)


def bbox_pcd(pc_path, visualize=False, visualize_by_cat=False):
    # Create a list to store all bounding box geometries, centroids, and point counts
    bounding_boxes = []
    centroids = []
    point_counts = []
    lbs = []
    cat_instances = defaultdict(list)

    # cluster min_points, object min_points, object max_points
    ceil = float('inf')
    bound_dict = {
        0: (50, 100, 150),  # 'Bollard',
        1: (120, 120, ceil),  # 'Building',
        2: (100, 100, 150),  # 'Bus Stop',
        3: (70, 100, 150),  # 'Control Box',
        4: (250, 300, ceil),  # 'Ground',
        5: (60, 80, 250),  # 'Lamp Post',
        6: (100, 100, 150),  # 'Pole'
        7: (70, 50, 100),  # 'Railing'
        8: (250, 250, ceil),  # 'Road',
        9: (150, 150, 250),  # 'Shrub',
        10: (50, 50, 100),  # 'Sign',
        11: (100, 100, 150),  # 'Solar Panel',
        12: (150, 200, ceil),  # 'Tree'
    }

    ############
    # read file #
    ############
    las_data = laspy.read(pc_path)
    # all_points = np.vstack((las_data.x, las_data.y, las_data.z)).T
    labels = las_data.pred  # #Classification
    # max bbox vol
    all_points = np.column_stack((las_data.x, las_data.y, las_data.z))
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    max_dimensions = max_coords - min_coords

    for tag in sorted(set(labels)):
        print('tag', tag, 'class', name_dict[tag])

        class_points = las_data.points[las_data.pred == tag]
        points = np.vstack((class_points.x, class_points.y, class_points.z)).T
        # Create a PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        #############
        # clustering #
        #############

        # Downsampling (xyz)
        pcd = pcd.voxel_down_sample(voxel_size=0.1)

        # Convert point cloud to numpy array
        points = np.asarray(pcd.points)

        # DBSCAN Cluster the points
        # clusters = np.array(pcd.cluster_dbscan(eps=0.5, min_points=bound_dict[tag][0], print_progress=True))
        clusters = np.array(pcd.cluster_dbscan(eps=0.5, min_points=10, print_progress=True))

        # Get unique clusters
        unique_cluster = np.unique(clusters)

        # Create bounding boxes for each cluster
        for cluster in unique_cluster:
            cluster_indices = np.where(clusters == cluster)[0]
            cluster_points = points[cluster_indices, :]

            # filter objects to checking whether a complete object, not applicable for downsampled
            # if len(cluster_points) >= label_to_min_points[tag]:

            # Compute bounding box
            min_coords = np.min(cluster_points, axis=0)
            max_coords = np.max(cluster_points, axis=0)

            # Compute centroid
            # centroid = np.mean(cluster_points, axis=0)
            centroid = get_center_base_coords(points, tag)

            # Get point count
            point_count = len(cluster_points)

            # Create bounding box geometry
            bbox = create_bounding_box(min_coords, max_coords)

            # Add bounding box, centroid, and point count to the respective lists
            # Manually remove large box for some objects
            dimensions = max_coords - min_coords
            # volume = dimensions[0] * dimensions[1] * dimensions[2]
            if tag not in (4, 8) and np.any(dimensions > 0.6 * max_dimensions):
                continue
            # Manually remove small box for some objects
            elif tag in (1, 4, 8):
                area = np.asarray(cluster_points)[:, 0:2]
                cat_instances[tag].append((min_coords, max_coords, centroid, area))
            else:
                cat_instances[tag].append((min_coords, max_coords, centroid))

            bounding_boxes.append(bbox)
            centroids.append(centroid)
            point_counts.append(point_count)
            lbs.append(tag)

        #############
        # visualize #
        #############
        if visualize_by_cat:
            all_geometries = [pcd]
            color = tuple(np.random.rand(3))
            for coords in cat_instances[tag]:
                min_coords, max_coords = coords[0], coords[1]
                bbox = create_bounding_box(min_coords, max_coords)
                bbox.paint_uniform_color(color)
                all_geometries.append(bbox)
            o3d.visualization.draw_geometries(all_geometries)

    #############
    # visualize #
    #############
    # full point cloud
    fpcd = o3d.geometry.PointCloud()
    fpcd.points = o3d.utility.Vector3dVector(all_points)

    # Visualize the original point cloud along with all bounding boxes
    if visualize:
        o3d.visualization.draw_geometries([fpcd] + bounding_boxes)

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
    try:
        pred_path = os.path.join("av_randlanet_scfnet/results/", filename, "predictions", filename)
        bbox_dict = bbox_pcd(pred_path, visualize=False)
        out_folder = os.path.join("av_randlanet_scfnet/results/", filename, "shapefiles_v3")
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
    except Exception as er:
        print(er)
        traceback.print_exc()


if __name__ == '__main__':
    # convert_main("Orchard_0913_labelled_E.laz")
    try:
        convert_main(sys.argv[1])
    except Exception as err:
        print(err)
