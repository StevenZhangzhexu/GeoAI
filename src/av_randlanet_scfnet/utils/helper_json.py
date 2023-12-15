import os
import json
import numpy as np


def get_base_center_coord(coords):
    # Assuming both clouds are in the same coordinate system (e.g., meters),
    # calculate the distance between centroids of both point clouds
    centroid = np.mean(coords, axis=0)
    base_point = coords[np.argmin(coords[:, 2])]
    base_centroid = centroid.copy()
    base_centroid[2] = base_point[2]
    print("centroid:", centroid, ", base_point:",
          base_point, ", base_centroid:", base_centroid)

    return base_centroid


def draw_line_and_get_points(x1, y1, x2, y2, resolution=100):
    """
      Draws a line between two points and returns all the points on the line with a specified resolution.

      Args:
        x1, y1: coordinates of the first point
        x2, y2: coordinates of the second point
        resolution: number of points to generate along the line (default 100)

      Returns:
        A list of tuples representing the coordinates of all the points on the line.
      """

    # Calculate delta x and delta y
    dx = x2 - x1
    dy = y2 - y1

    # Ensure non-zero step size
    step_size = max(abs(dx), abs(dy)) / resolution

    # Check for overlapping points
    if step_size == 0:
        print("Points are overlapping, cannot draw line.")
        return []

    # Initialize empty list to store points
    points = []

    # Iterate over each step and calculate corresponding x and y values
    for i in range(resolution + 1):
        x = x1 + i * dx / resolution
        y = y1 + i * dy / resolution
        points.append((x, y))
        print([x, y])

    # Return the list of points
    return points


def get_closest_road_center(object_center):
    middle_points_3d = []
    # main orchard road @TODO: get start and end of the road automatically
    middle_points = draw_line_and_get_points(
        103.83647747, 1.30250373, 103.83390328, 1.30384428, resolution=100)
    for each in middle_points:
        t = [each[0], each[1], object_center[2]]
        middle_points_3d.append(t)
    middle_points_3d = np.asarray(middle_points_3d)
    # print(middle_points_3d)

    min_dist = 99999999999
    closest_point = None
    for each in middle_points_3d:
        dist = np.linalg.norm(each - object_center)
        # print(each, dist)
        if dist < min_dist:
            closest_point = each
            min_dist = dist

    print("Closest point:", closest_point, " , distance:", min_dist)
    return closest_point


def get_axes_orientation_by_unit_vector(point1, point2):
    # Calculate the vector from point1 to point2
    vector = point2 - point1

    # Normalize the vector to get the unit vector
    unit_vector = vector / np.linalg.norm(vector)

    # Output the unit vector representing the orientation
    print("Unit Vector:", unit_vector)
    return unit_vector


def get_center_base_coord(coordinates):
    # Find the point with the lowest z-coordinate (assuming z represents height)
    base_point = coordinates[np.argmin(coordinates[:, 2])]
    center_coordinate = np.mean(coordinates, axis=0)

    base_center_coord = {
        'x': center_coordinate[0],
        'y': center_coordinate[1],
        # 'z': center_coordinate[2],
        'z': base_point[2],
    }
    print(base_center_coord)

    return base_center_coord


def get_start_end_center_base_coords(coordinates):
    # Find the point with the lowest z-coordinate (assuming z represents height)
    base_point = coordinates[np.argmin(coordinates[:, 2])]
    start_coordinate = np.min(coordinates, axis=0)
    end_coordinate = np.max(coordinates, axis=0)
    center_coordinate = np.mean(coordinates, axis=0)

    bc = get_base_center_coord(coordinates)
    cc = get_closest_road_center(bc)
    orientations = get_axes_orientation_by_unit_vector(bc, cc)

    # base_center_coord = {
    #         'x': center_coordinate[0],
    #         'y': center_coordinate[1],
    #         # 'z': center_coordinate[2],
    #         'z': base_point[2],
    # }
    # print(base_center_coord)

    start = {
        'x': start_coordinate[0],
        'y': start_coordinate[1],
        'z': base_point[2],
    }
    end = {
        'x': end_coordinate[0],
        'y': end_coordinate[1],
        'z': base_point[2],
    }
    center = {
        'x': center_coordinate[0],
        'y': center_coordinate[1],
        'z': base_point[2],
    }
    orientation = {
        'x': orientations[0],
        'y': orientations[1],
        'z': orientations[2]
    }

    return start, end, center, orientation


def get_road_coordinates(coordinates):
    # Get the first, last, and middle coordinates
    first_point = coordinates[0]
    last_point = coordinates[-1]
    middle_point = coordinates[len(coordinates) // 2]  # Calculate middle index

    return first_point, last_point, middle_point


def get_start_end_middle_coordinates(coordinates):
    # Calculate the length of the point cloud
    num_points = len(coordinates)

    # Define the percentage of the road to consider for start and end
    start_percentage = 0.1
    end_percentage = 0.9

    # Calculate indices for start, end, and middle coordinates
    start_index = int(num_points * start_percentage)
    end_index = int(num_points * end_percentage)
    middle_index = num_points // 2

    # Extract start, end, and middle coordinates
    start_coord = coordinates[start_index]
    end_coord = coordinates[end_index]
    middle_coord = coordinates[middle_index]

    return start_coord, end_coord, middle_coord


def save_objects_json(output_file_path, objects):
    with open(output_file_path, 'w') as json_file:
        json.dump(objects, json_file, indent=4)
