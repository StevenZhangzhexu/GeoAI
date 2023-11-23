import os
import json
import numpy as np


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
    # print(base_center_coord)

    return base_center_coord


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

