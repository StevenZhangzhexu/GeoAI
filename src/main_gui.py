import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.filechooser import FileChooserIconView

# Replace with your actual deep learning model and functions
from av_randlanet_scfnet import predict_OrchardRoad
from av_randlanet_scfnet.utils import data_prepare_orchard, separate_predicted_objects, helper_las


class PointCloudSegmentationApp(App):
    file_path = StringProperty("")
    progress_text = StringProperty("No file selected.")
    segmented_cloud = ObjectProperty(None)  # Placeholder for storing segmented point cloud

    def build(self):
        layout = BoxLayout(orientation="vertical")

        # File path label
        file_path_label = Label(text="Point Cloud File:")
        layout.add_widget(file_path_label)

        # File chooser button
        file_chooser_button = Button(text="Browse...")
        file_chooser_button.bind(on_press=self.open_file_chooser)
        layout.add_widget(file_chooser_button)

        # File path display
        self.file_path_input = TextInput(hint_text="Selected File Path", multiline=False, readonly=True)
        layout.add_widget(self.file_path_input)

        # Submit button
        submit_button = Button(text="Submit for Segmentation")
        submit_button.bind(on_press=self.segment_point_cloud)
        layout.add_widget(submit_button)

        # Progress label
        self.progress_label = Label(text=self.progress_text)
        layout.add_widget(self.progress_label)

        # Placeholder for displaying segmented point cloud (replace with your visualization method)
        self.segmented_cloud_label = Label(text="Segmented Point Cloud:")
        layout.add_widget(self.segmented_cloud_label)

        return layout

    def open_file_chooser(self, instance):
        file_chooser = FileChooserIconView()
        file_chooser.bind(on_selection=self.set_file_path)  # Bind on_selection event
        file_chooser.open()

    def set_file_path(self, instance, value):
        if value:
            self.file_path = value[0]  # Access the first selected file path
            self.file_path_input.text = self.file_path
        else:
            self.file_path = ""
            self.file_path_input.text = "No file selected."

    def segment_point_cloud(self, instance):
        if not self.file_path:
            self.progress_text = "Error: Please enter a file path."
            self.update_progress_label()
            return

        # Simulate segmentation process (replace with your actual deep learning calls)
        self.progress_text = "Loading point cloud..."
        self.update_progress_label()
        # pre-process
        data_prepare_orchard.prepare_data(self.file_path)

        # predict
        predict_OrchardRoad.predict(filepath=self.file_path)

        # post-process
        filename = self.file_path.split("/")[-1]
        # separate_predicted_objects.separate_segmented_point_clouds(filename)
        # separate_predicted_objects.separate_and_cluster_point_clouds(filename)
        separate_predicted_objects.separate_and_segment_point_clouds(filename)

        self.progress_text = "Segmenting point cloud..."
        self.update_progress_label()
        # self.segmented_cloud = segment_point_cloud(point_cloud)  # Replace with actual segmentation

        self.progress_text = "Segmentation complete."
        self.update_progress_label()

        # Update UI to display segmented point cloud (replace with your visualization method)
        # This example just shows a placeholder message
        self.segmented_cloud_label.text = "Segmented Point Cloud: \n(Replace this with your visualization method)"

    def update_progress_label(self):
        self.progress_label.text = self.progress_text


if __name__ == "__main__":
    PointCloudSegmentationApp().run()
