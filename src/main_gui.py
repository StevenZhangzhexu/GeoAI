import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import time
import laspy
import open3d as o3d
import numpy as np
from av_randlanet_scfnet import predict_OrchardRoad
from av_randlanet_scfnet.utils import data_prepare_orchard, separate_predicted_objects, helper_las


# Global variable to store the selected file path
file_path = None


# Define a colormap for labels
def label_to_color(label):
    # Define your colormap here
    colormap = [
        # [0, 0, 0],     # Label 0 (background) as black
        [255, 0, 0],   # Label 1 as red
        [0, 255, 0],   # Label 2 as green
        [0, 0, 255],   # Label 3 as blue
        [255, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [0, 128, 255],
        [255, 128, 0],
        [0, 255, 128],
        [0, 128, 0],
        [0, 0, 128],
        [128, 0, 0],
        [128, 128, 0],
        [128, 0, 128],
        [128, 128, 0],
        # Add more colors for additional labels as needed
    ]
    if label < len(colormap):
        return colormap[label]
    else:
        return [255, 255, 255]  # Default to white for unknown labels


# Function to create a progress bar with message
def create_progressbar(root, message):
    progressbar = ttk.Progressbar(root, orient="horizontal", mode="indeterminate")
    progressbar.pack(pady=10)
    label = tk.Label(root, text=message)
    label.pack()
    return progressbar


# Function to open file dialog and load selected point cloud
def open_file_dialog():
    global file_path
    file_types = [("LAZ Files", "*.laz"), ("LAS Files", "*.las")]
    file_path = filedialog.askopenfilename(filetypes=file_types)
    if file_path:
        print("Selected file:", file_path)
        visualize_point_cloud(file_path)


def o3d_viz(cloud, predicted=False):
    points = np.vstack((cloud.x, cloud.y, cloud.z)).T

    # Create an Open3D PointCloud object with colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if predicted:
        predictions = cloud.pred
        print(np.unique(predictions))
        predictions = cloud.classification
        print(np.unique(cloud.classification))
        colors = np.array([label_to_color(label) for label in predictions])  # Map labels to colors
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0, 1]

    # Visualize the merged point cloud with colors
    o3d.visualization.draw_geometries([pcd])


# Function to visualize point cloud
def visualize_point_cloud(file_path, annotated=False):
    # Load point cloud data using laspy
    las_data = laspy.read(file_path)
    o3d_viz(las_data, predicted=annotated)


# Function to perform point cloud segmentation
def perform_segmentation():
    global file_path
    if file_path:
        progressbar = create_progressbar(root, "Segmenting point cloud...")

        # (Replace with actual segmentation code)
        time.sleep(5)  # Simulate segmentation process (5 seconds)

        # pre-process
        data_prepare_orchard.prepare_data(file_path)

        # predict
        predict_OrchardRoad.predict(filepath=file_path)

        # post-process
        filename = file_path.split("/")[-1]
        # separate_predicted_objects.separate_segmented_point_clouds(filename)
        # separate_predicted_objects.separate_and_cluster_point_clouds(filename)
        #separate_predicted_objects.separate_and_segment_point_clouds(filename)

        segmented_point_cloud_file_path = "av_randlanet_scfnet/results/" + filename + "/predictions/" + filename

        progressbar.destroy()  # Remove progress bar after segmentation

        # vis results
        visualize_point_cloud(segmented_point_cloud_file_path, annotated=True)


# Create main window
root = tk.Tk()
root.title("Point Cloud Segmentation")

# Set window size
root.geometry("800x600")  # Width x Height

# Create frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=20)

# Create button to open file dialog
btn_open = tk.Button(button_frame, text="Open Point Cloud", command=open_file_dialog)
btn_open.grid(row=0, column=0, padx=10)

# Create button to perform segmentation
btn_submit = tk.Button(button_frame, text="Submit", command=perform_segmentation)
btn_submit.grid(row=0, column=1, padx=10)

# Run the GUI
root.mainloop()
