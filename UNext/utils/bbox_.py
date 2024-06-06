import open3d as o3d
import numpy as np
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

def bbox_pcd(las_data, name_dict, visualize = False, visualize_by_cat = False, restore=False, bbox=None, skip=set()):
    # Create a list to store all bounding box geometries, centroids, and point counts
    bounding_boxes = []
    centroids = []
    point_counts = []
    lbs=[]
    vol=[]
    cat_instances = bbox if bbox else defaultdict(list)
    
    # cluster min_points, object min_points, object max_points
    bound_dict = {
                0: 100,
                1: 60,
                2: 50,
                3: 100,
                4: 100,
                5: 100,
                6: 100,
                7: 100,
                8: 100,
                9: 100,
                10: 100,
                11: 100,
                12: 100,
                13: 100,
                14: 100,
                15: 100,
                16: 100,
                17: 100,
                18: 100,
                19: 100,
                20: 100,
                21: 100,
                22: 100
                }

    ############
    # input pcd #
    ############
    # all_points = np.vstack((las_data.x, las_data.y, las_data.z)).T
    labels = las_data.pred #   #Classification
    tags = set(labels) - skip
    # max bbox vol
    all_points = np.column_stack((las_data.x, las_data.y, las_data.z))
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    max_dimensions = max_coords - min_coords
    threshold_z = 0.3 * max_dimensions[2] # maxz
    mask = las_data.z < threshold_z
    
    for tag in sorted(tags):
        if name_dict.get(tag) == 'Unclassified'  or (tag not in name_dict):
            continue
        print('tag',tag, 'class', name_dict[tag])
        if name_dict[tag] in ('Building', 'BusStop', 'Ground', 'Railing', 'Road', 'Tree', 'Hydrant', 'Shed', 'Overpass','PedestrianOverheadBridge', 'Barrier'):
            epsilon = 1.2
        elif name_dict[tag] in ('Bollard', 'Pole', 'Sign', 'Control Box'):
            epsilon = 0.8      
        else:
            epsilon = 0.5

        if name_dict[tag] in ('Building', 'BusStop', 'Ground', 'Road', 'Shed', 'TrafficLight', 'Overpass','PedestrianOverheadBridge'): 
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
        clusters = np.array(pcd.cluster_dbscan(eps = epsilon, min_points = bound_dict[tag], print_progress=True))

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
            # bbox = create_bounding_box(min_coords, max_coords)

            # Add bounding box, centroid, and point count to the respective lists
            # Menually remove large box for some objects
            dimensions = max_coords - min_coords
            volume = dimensions[0] * dimensions[1] * dimensions[2]

            # if name_dict[tag] not in ('Ground', 'Road') and np.any(dimensions > 0.5 * max_dimensions):
            #     continue
            if name_dict[tag] not in ('Building', 'Ground', 'Railing', 'Road','Shed', 'Overpass','PedestrianOverheadBridge', 'Barrier','CoveredLinkway','Pathway') and volume > 500 :
                continue
            # elif name_dict[tag] == 'Tree' and max_coords[2] > threshold_z:
            #     continue
            elif name_dict[tag] == 'ControlBox'  and (point_count<200 and volume<1):
                continue
            # Menually remove small box for some objects
            elif name_dict[tag] in ('Building', 'BusStop', 'Ground', 'Road', 'Bollard', 'Pole', 'Railing', 'Hydrant', 'Shed', 'Overpass', 'PedestrianOverheadBridge', 'ZebraBeaconPole', 'Barrier','CoveredLinkway','Pathway'):
                if point_count <2:
                    continue
                area = np.asarray(cluster_points)[:,0:2]
                cat_instances[tag].append((min_coords, max_coords, centroid, area))
            else:
                cat_instances[tag].append((min_coords, max_coords, centroid))

            # bounding_boxes.append(bbox)
            centroids.append(centroid)
            point_counts.append(point_count)
            lbs.append(tag)
            vol.append(volume)

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
    # Visualize the original point cloud along with all bounding boxes
    if visualize:
        fpcd = o3d.geometry.PointCloud()
        fpcd.points = o3d.utility.Vector3dVector(all_points)
        o3d.visualization.draw_geometries([fpcd] + bounding_boxes)

    # Print centroids and point counts
    for i, centroid in enumerate(centroids):
        print(f"Box {i+1} - Centroid: {centroid}, Point Count: {point_counts[i]}. Size: {vol[i]}, Label: {lbs[i]}")

    print(cat_instances.keys())
         
    return cat_instances



