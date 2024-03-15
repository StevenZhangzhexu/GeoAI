from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
import time
from av_randlanet_scfnet import predict_OrchardRoad
from av_randlanet_scfnet.utils import data_prepare_orchard, separate_predicted_objects, helper_las


# Placeholder function to segment point cloud (replace with actual segmentation code)
def segment_point_cloud(file_path):
    time.sleep(5)  # Simulate segmentation process (5 seconds)
    # segmented_point_cloud = file_path + "_segmented"  # Placeholder for segmented point cloud file path

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

    return segmented_point_cloud_file_path


# Placeholder function to visualize segmented point cloud (replace with actual visualization code)
def visualize_segmented_point_cloud(segmented_point_cloud_path):
    print("Visualizing segmented point cloud:", segmented_point_cloud_path)


class FileChooserApp(App):
    def build(self):
        # Create the main layout
        self.layout = BoxLayout(orientation="vertical")

        # Button to trigger the file chooser
        self.button = Button(text="Open Point Cloud", on_press=self.open_file_chooser)
        self.layout.add_widget(self.button)

        # Label to display the selected file path (optional)
        self.file_path_label = Label(text="No file selected")
        self.layout.add_widget(self.file_path_label)

        # Progress bar
        self.progress_bar = ProgressBar(max=100)
        self.layout.add_widget(self.progress_bar)

        return self.layout

    def open_file_chooser(self, instance):
        # Create the file chooser object
        file_chooser = FileChooserListView()

        # Bind the selection event to perform segmentation
        file_chooser.bind(selection=self.segment_selected_file)

        # Open the file chooser dialog
        self.popup = Popup(title='Select a Point Cloud File', content=file_chooser, size_hint=(0.9, 0.9))
        self.popup.open()

    def segment_selected_file(self, instance, selection):
        if selection:
            # Get the selected file path
            file_path = selection[0]
            print(file_path)

            # Update the label text with the selected file path
            self.file_path_label.text = "Selected file: " + file_path

            # Close the file chooser popup
            self.popup.dismiss()

            # Perform point cloud segmentation
            segmented_point_cloud_path = segment_point_cloud(file_path)

            # Visualize segmented point cloud
            visualize_segmented_point_cloud(segmented_point_cloud_path)
        else:
            # No file selected, reset label text
            self.file_path_label.text = "No file selected"


if __name__ == "__main__":
    FileChooserApp().run()
