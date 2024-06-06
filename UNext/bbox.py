import open3d as o3d
import numpy as np
import laspy
# from scipy.spatial import ConvexHull
# from scipy import stats
from collections import defaultdict

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

def bbox_pcd(pc_path, visualize = True, visualize_by_cat = False):
    # Create a list to store all bounding box geometries, centroids, and point counts
    bounding_boxes = []
    centroids = []
    point_counts = []
    lbs=[]
    cat_instances = defaultdict(list)

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
    
    # cluster min_points, object min_points, object max_points
    ceil = float('inf')
    bound_dict = {
                    0: (50, 100, 150),  #'Bollard',
                    1: (120, 120, ceil),  #'Building',
                    2: (100, 100, 150),  #'Bus Stop',
                    3: (70, 100, 150),  #'Control Box',
                    4: (250, 300, ceil), #'Ground',
                    5: (60, 80, 250),  #'Lamp Post',
                    6: (100, 100, 150),  #'Pole' 
                    7: (70, 50, 100),    #'Railing'
                    8: (250, 250, ceil), #'Road',
                    9: (150, 150, 250),  #'Shrub',
                    10: (50, 50, 100),   #'Sign',
                    11: (100, 100, 150), #'Solar Panel',
                    12: (150, 200, ceil), #'Tree'
                }

    ############
    # read file #
    ############
    las_data = laspy.read(pc_path)
    # all_points = np.vstack((las_data.x, las_data.y, las_data.z)).T
    labels = las_data.pred #   #Classification
    # max bbox vol
    all_points = np.column_stack((las_data.x, las_data.y, las_data.z))
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    max_dimensions = max_coords - min_coords
    threshold_z = 0.3 * max_dimensions[2] # maxz
    mask = las_data.z < threshold_z
    

    for tag in sorted(set(labels)):
        print('tag',tag, 'class', name_dict[tag])
        if tag in (1,2,4,8,12):
            epsilon = 1.2
        # elif tag in (10,):
        #     epsilon = 1
        elif tag in (0,6,10):
            epsilon = 0.8      
        else:
            epsilon = 0.5

        if tag in (1,2,4,8):
            class_points = las_data.points[las_data.pred == tag]
        else:
            class_points = las_data.points[(las_data.pred == tag) & mask]
         
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
        clusters = np.array(pcd.cluster_dbscan(eps = epsilon, min_points = bound_dict[tag][0], print_progress=True))

        # Get unique clusters
        unique_cluster = np.unique(clusters)


        # Create bounding boxes for each cluster
        for cluster in unique_cluster:
            cluster_indices = np.where(clusters == cluster)[0]
            cluster_points = points[cluster_indices, :]

            # Compute bounding box
            min_coords = np.min(cluster_points, axis=0)
            max_coords = np.max(cluster_points, axis=0)

            # Compute centroid
            centroid = np.mean(cluster_points, axis=0)

            # Get point count
            point_count = len(cluster_points)

            # Create bounding box geometry
            bbox = create_bounding_box(min_coords, max_coords)

            # Add bounding box, centroid, and point count to the respective lists
            # Menually remove large box for some objects
            dimensions = max_coords - min_coords
            # volume = dimensions[0] * dimensions[1] * dimensions[2]
            if tag not in (4, 8) and np.any(dimensions > 0.5 * max_dimensions):
                continue
            elif tag==12 and max_coords[2] > threshold_z:
                continue
            # Menually remove small box for some objects
            elif tag in (1,2,4,8, 0,6):
                area = np.asarray(cluster_points)[:,0:2]
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
        print(f"Box {i+1} - Centroid: {centroid}, Point Count: {point_counts[i]}. Label: {lbs[i]}")

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

