import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import time

from av_randlanet_scfnet import predict_OrchardRoad
from av_randlanet_scfnet.utils import data_prepare_orchard, separate_predicted_objects, helper_las


# Global variable to store the selected file path
file_path = None


# Function to create a progress bar with message
def create_progressbar(root, message):
    progressbar = ttk.Progressbar(root, orient="horizontal", mode="indeterminate")
    progressbar.pack(pady=10)
    progressbar.start()
    label = tk.Label(root, text=message)
    label.pack()
    return progressbar


# Function to open file dialog and load selected point cloud
def open_file_dialog():
    global file_path
    file_types = [("LAS Files", "*.las"), ("LAZ Files", "*.laz"), ("PCD Files", "*.pcd")]
    file_path = filedialog.askopenfilename(filetypes=file_types)
    if file_path:
        print("Selected file:", file_path)
        # (Replace with actual point cloud loading and visualization code)
        print("Loading and visualizing point cloud...")


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
        # separate_predicted_objects.separate_and_segment_point_clouds(filename)

        segmented_point_cloud_path = "av_randlanet_scfnet/results/" + filename + "/predictions/" + filename

        progressbar.destroy()  # Remove progress bar after segmentation

        # (Replace with actual visualization of segmented point cloud)
        print("Visualizing segmented point cloud:", segmented_point_cloud_path)

# Create main window
root = tk.Tk()
root.title("Point Cloud Segmentation")

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
