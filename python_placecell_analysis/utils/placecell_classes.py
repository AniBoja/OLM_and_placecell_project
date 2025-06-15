import ipywidgets as widgets
from IPython.display import display, clear_output
from scipy.io import loadmat
import pandas as pd


# create a dataloader class that will handel loading the data and processing names on the radio button changes
class data_loader:
    def __init__(self, data_dir, exp_sets, data_sets, data_filenames, data_titles, session_names):
        self.data_dir = data_dir
        self.exp_sets = exp_sets
        self.data_sets = data_sets
        self.data_filenames = data_filenames
        self.data_titles = data_titles
        self.session_names = session_names
        self.observers = []  # Store observers to notify them when data changes
        self.create_widget()
        self.exp_widget.observe(self.update_data, names="value")
        self.data_widget.observe(self.update_data, names="value")
        self.update_data("")

    def create_widget(self):
        self.data_widget = widgets.RadioButtons(
            options=self.data_sets, index=0, description="Dataset:", disabled=False)
        self.exp_widget = widgets.RadioButtons(
            options = self.exp_sets, index = 0, description="Experiments:", disabled=False)
        

    def show_widget(self):
        display(self.exp_widget)
        display(self.data_widget)
    
    def add_observer(self, observer_callback):
        """Allow external components to subscribe to data changes."""
        self.observers.append(observer_callback)
        
    def notify_observers(self):
        """Notify all registered observers that data has changed."""
        for callback in self.observers:
            callback(self)  # Pass the current data_loader instance to observers


    def update_data(self, event):
        
        self.data_exp_dir = f"{self.data_dir}{self.exp_widget.value}/"
        
        self.data_set_to_plot = self.data_filenames[
            self.data_sets.index(self.data_widget.value)
        ]
        self.sub_titles = self.session_names[
            self.data_sets.index(self.data_widget.value)
        ]
        self.title_for_data = self.data_titles[
            self.data_sets.index(self.data_widget.value)
        ]

        self.placecell_rates_dict, self.raw_rates_dict, self.placecell_info_dict, self.spikes_dict, self.behaviour_dict = (
            self.read_data_file(self.data_set_to_plot)
        )

        self.animals = list(self.placecell_rates_dict.keys())
        self.animals_options = self.animals.copy()
        self.animals_options.append("pooled data")
        # print("data updated")

        # Notify all observers about the data change
        self.notify_observers()  # This triggers animal_widgets to update

    def read_data_file(self, filename):
        file_dir = self.data_exp_dir + filename
        mat_data = loadmat(file_dir, simplify_cells=True)
        rates_dict = mat_data["rates_structure"]

        def create_ratemap_dict(rates_dict):
            animals = list(rates_dict.keys())  # get the animal names
            sessions = list(
                rates_dict[animals[0]].keys()
            )  # this assumes that all animals have the same session names

            placecell_rates = {}
            for mouse in animals:
                session_rates = []
                for sesh in sessions:
                    rates = pd.DataFrame(rates_dict[mouse][sesh]["rates"])
                    ids = rates_dict[mouse][sesh]["rates_IDS"]
                    rates.index = ids
                    session_rates.append(rates)

                placecell_rates[mouse] = session_rates
            return placecell_rates

        def create_info_dict(info_dict):
            animals = list(info_dict.keys())  # get the animal names
            sessions = list(
                info_dict[animals[0]].keys()
            )  # this assumes that all animals have the same session names

            placecell_info = {}
            for mouse in animals:
                session_info = []
                for sesh in sessions:
                    info = pd.DataFrame(info_dict[mouse][sesh]["spatial_info"])
                    ids = info_dict[mouse][sesh]["cell_IDS"]
                    info.index = ids
                    session_info.append(info)

                placecell_info[mouse] = session_info
            return placecell_info
        
        def create_spike_dict(info_dict):
            animals = list(info_dict.keys())  # get the animal names
            sessions = list(
                info_dict[animals[0]].keys()
            )  # this assumes that all animals have the same session names

            spike_data = {}
            for mouse in animals:
                session_info = []
                for sesh in sessions:
                    sesh_dict = {}
                    spikes = pd.DataFrame(info_dict[mouse][sesh]["spike_times"])
                    spikes.columns = info_dict[mouse][sesh]["cell_ids"]
                    
                    sesh_dict['spike times'] = spikes
                    sesh_dict['is_placecell'] = info_dict[mouse][sesh]["is_placecell"]

                    session_info.append(sesh_dict)

                spike_data[mouse] = session_info
            return spike_data
        
        def create_behvaiour_dict(info_dict):
            animals = list(info_dict.keys())  # get the animal names
            sessions = list(
                info_dict[animals[0]].keys()
            )  # this assumes that all animals have the same session names

            behave_data = {}
            for mouse in animals:
                session_info = []
                for sesh in sessions:
                    behaviour = info_dict[mouse][sesh]
                    session_info.append(behaviour)

                behave_data[mouse] = session_info
            return behave_data

        placecell_rates_dict = create_ratemap_dict(rates_dict["placecell_rates"])
        raw_rates_dict = create_ratemap_dict(rates_dict["raw_rates"])

        placecell_info_dict = create_info_dict(rates_dict["placecell_info"])
        spikes_dict = create_spike_dict(rates_dict['spikes'])
        behaviour_dict = create_behvaiour_dict(rates_dict['behaviour'])


        return (
            placecell_rates_dict,
            raw_rates_dict,
            placecell_info_dict,
            spikes_dict,
            behaviour_dict
        )  # , allcell_info_dict


class animal_widgets:
    def __init__(self, output_widget, data):
        self.output_widget = output_widget
        self.data = data
        # self.create_widget(self.output_widget)
        # Register this widget to listen for updates from data_loader
        self.data.add_observer(self.update_animal_options)

    def create_widget(self, output_widget):

        self.output_widget = output_widget
        # create a drop down widget to select pooled data or an indiviual mouse
        self.selection = widgets.Dropdown(
            options=self.data.animals_options,
            index=len(self.data.animals_options) - 1,
            description="Which animal?",
            style={"description_width": "initial"},
            disabled=False,
        )
        self.selection.layout.margin = "10px"

        # create a tags widget to select which animals to include in the pooled data
        self.ids_to_pool = widgets.TagsInput(
            value=self.data.animals,
            allowed_tags=self.data.animals,
            allow_duplicates=False,
            description="Animals to analyse",
            style={"description_width": "400px"},
        )
        self.ids_to_pool.layout.width = "250px"
        self.ids_to_pool.layout.flex_flow = "wrap"

        # observe the changes for the animal ID drop down
        self.selection.observe(self.on_value_change, names="value")

        self.output_widget.clear_output()
        with self.output_widget:
            display(self.selection)
            display(self.ids_to_pool)

        return self.output_widget
    
    
    def update_animal_options(self, data):
        """Callback to update the widget options when data changes."""
        self.data = data
        self.selection.options = data.animals_options  # Update Dropdown options
        self.ids_to_pool.allowed_tags = data.animals   # Update allowed tags
        self.ids_to_pool.value = data.animals          # Update TagsInput value

        self.create_widget(self.output_widget)  # Recreate the widget with updated values


    # A function to update the output widget depending on if its pooled data or individual data
    def on_value_change(self, change):
        if self.selection.value == "pooled data":
            with self.output_widget:
                clear_output()
                display(self.selection)
                display(self.ids_to_pool)  # display the animal selection
        else:
            with self.output_widget:
                clear_output()  # Clear the output
                display(self.selection)
                new_image_path = f"{self.data.data_dir}FOV/{self.selection.value}.png"  # Replace with your new image path
                display(f"{self.selection.value} field of view")
                new_image_widget = widgets.Image(
                    value=open(new_image_path, "rb").read()
                )
                display(new_image_widget)  # display the field of view for the mouse
