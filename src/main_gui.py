import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import laspy
import pyvista as pv


# Global variable to store the selected file path
file_path = None


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


# Function to visualize point cloud
def visualize_point_cloud(file_path):
    # Load point cloud data using laspy
    las_data = laspy.read(file_path)
    points = las_data.points

    # Extract x, y, z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create a PyVista point cloud
    cloud = pv.PolyData(points)

    # Plot the point cloud using PyVista
    plotter = pv.Plotter()
    plotter.add_points(cloud, color="blue", point_size=2)
    plotter.show()


# Function to perform point cloud segmentation
def perform_segmentation():
    global file_path
    if file_path:
        progressbar = create_progressbar(root, "Segmenting point cloud...")

        # (Replace with actual segmentation code)
        time.sleep(5)  # Simulate segmentation process (5 seconds)

        progressbar.destroy()  # Remove progress bar after segmentation


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
