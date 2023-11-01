import numpy as np
import cv2

def project_image_onto_sphere(image):
  """Projects an image onto a sphere.

  Args:
    image: A NumPy array representing the input image.

  Returns:
    A NumPy array representing the projected image.
  """

  # Get the image dimensions.
  height, width = image.shape[:2]

  # Create a sphere.
  sphere = np.zeros((height, width, 3), dtype=np.float32)

  # Project the image onto the sphere.
  for i in range(height):
    for j in range(width):
      # Get the pixel coordinates in the input image.
      pixel_coordinates = (j, i)

      # Get the corresponding spherical coordinates.
      spherical_coordinates = project_pixel_onto_sphere(pixel_coordinates, height, width)

      # Set the pixel color in the output image.
      sphere[i, j] = image[spherical_coordinates[1], spherical_coordinates[0]]

  return sphere

def project_pixel_onto_sphere(pixel_coordinates, height, width):
  """Projects a pixel in an image onto a sphere.

  Args:
    pixel_coordinates: A tuple representing the pixel coordinates in the input image.
    height: The height of the input image.
    width: The width of the input image.

  Returns:
    A tuple representing the spherical coordinates of the projected pixel.
  """

  # Get the pixel coordinates in the range [0, 1].
  normalized_pixel_coordinates = (pixel_coordinates[0] / width, pixel_coordinates[1] / height)

  # Calculate the spherical coordinates of the projected pixel.
  theta = normalized_pixel_coordinates[0] * 2 * np.pi
  phi = normalized_pixel_coordinates[1] * np.pi - np.pi / 2

  return (theta, phi)

# Load the input image.
image = cv2.imread("input.jpg")

# Project the image onto a sphere.
projected_image = project_image_onto_sphere(image)

# Save the projected image.
cv2.imwrite("output.jpg", projected_image)
