import numpy as np
import cv2
import open3d as o3d

def project_sphere_image_onto_point_cloud(sphere_image, point_cloud):
  """Projects a sphere image onto a point cloud.

  Args:
    sphere_image: A NumPy array representing the sphere image.
    point_cloud: An o3d.PointCloud() object representing the point cloud.

  Returns:
    An o3d.PointCloud() object representing the projected sphere image.
  """

  # Get the height and width of the sphere image.
  height, width = sphere_image.shape[:2]

  # Create a new point cloud to store the projected sphere image.
  projected_point_cloud = o3d.PointCloud()

  # Iterate over the pixels in the sphere image.
  for i in range(height):
    for j in range(width):
      # Get the pixel coordinates in the sphere image.
      pixel_coordinates = (j, i)

      # Get the corresponding spherical coordinates.
      spherical_coordinates = project_pixel_onto_sphere(pixel_coordinates, height, width)

      # Project the spherical coordinates onto the point cloud.
      projected_point = project_spherical_coordinates_onto_point_cloud(spherical_coordinates, point_cloud)

      # Add the projected point to the new point cloud.
      projected_point_cloud.points.append(projected_point)

  # Return the new point cloud.
  return projected_point_cloud

def project_pixel_onto_sphere(pixel_coordinates, height, width):
  """Projects a pixel in a sphere image onto a sphere.

  Args:
    pixel_coordinates: A tuple representing the pixel coordinates in the sphere image.
    height: The height of the sphere image.
    width: The width of the sphere image.

  Returns:
    A tuple representing the spherical coordinates of the projected pixel.
  """

  # Get the pixel coordinates in the range [0, 1].
  normalized_pixel_coordinates = (pixel_coordinates[0] / width, pixel_coordinates[1] / height)

  # Calculate the spherical coordinates of the projected pixel.
  theta = normalized_pixel_coordinates[0] * 2 * np.pi
  phi = normalized_pixel_coordinates[1] * np.pi - np.pi / 2

  return (theta, phi)

def project_spherical_coordinates_onto_point_cloud(spherical_coordinates, point_cloud):
  """Projects a spherical coordinates onto a point cloud.

  Args:
    spherical_coordinates: A tuple representing the spherical coordinates.
    point_cloud: An o3d.PointCloud() object representing the point cloud.

  Returns:
    An o3d.Vector3d() object representing the projected point.
  """

  # Calculate the Cartesian coordinates of the spherical coordinates.
  x = np.cos(spherical_coordinates[0]) * np.sin(spherical_coordinates[1])
  y = np.sin(spherical_coordinates[0]) * np.sin(spherical_coordinates[1])
  z = np.cos(spherical_coordinates[1])

  # Find the nearest point in the point cloud to the Cartesian coordinates.
  nearest_point = point_cloud.find_nearest_point([x, y, z])

  # Return the nearest point.
  return nearest_point

# Load the sphere image.
sphere_image = cv2.imread("sphere_image.jpg")

# Load the point cloud.
point_cloud = o3d.io.read_point_cloud("point_cloud.ply")

# Project the sphere image onto the point cloud.
projected_point_cloud = project_sphere_image_onto_point_cloud(sphere_image, point_cloud)

# Save the projected point cloud.
o3d.io.write_point_cloud("projected_point_cloud.ply", projected_point_cloud)
