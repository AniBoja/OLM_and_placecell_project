import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import h5py

from scipy.ndimage import uniform_filter1d, gaussian_filter
from scipy import stats


# Data Process functions


# might remove this one as its encorporated into the data loader
def sort_ratemaps(rates):

    maxrate_index = np.zeros(rates.shape[0], dtype=int)
    for i in range(rates.shape[0]):
        idx = np.argmax(rates.iloc[i, :])
        maxrate_index[i] = idx

    # Sort indices in ascending order
    id = np.argsort(maxrate_index)[::-1]
    idexnames = rates.index[id]

    return idexnames, maxrate_index


def sort_placecells(place_maps):
    placemaps_sorted = []
    for i in range(len(place_maps)):

        sort_id, _ = sort_ratemaps(place_maps[i])
        placemaps_sorted.append(place_maps[i].reindex(sort_id).fillna(0))

    return placemaps_sorted


def get_placecell_centers(place_maps):
    maxrate_index = np.zeros(place_maps.shape[0], dtype=int)
    for i in range(place_maps.shape[0]):
        idx = np.argmax(place_maps.iloc[i, :])
        maxrate_index[i] = idx
    return maxrate_index


def create_pooled_data(data_dict, keys_to_include):
    combined_dataframes = []

    for i in range(len(data_dict[keys_to_include[0]])):
        combined_df = pd.DataFrame()
        for key in keys_to_include:
            df = data_dict[key][i]
            combined_df = pd.concat([combined_df, df], axis=0)  # Concatenate verticle

        combined_dataframes.append(combined_df)

    return combined_dataframes


def get_pooled_average_rates(data_dict, pooled_key):
    pooled_rates_mean = pd.DataFrame()
    pooled_rates_sem = pd.DataFrame()
    sessions = ["session_1", "session_2", "session_3"]
    for session in range(len(data_dict[pooled_key[0]])):
        average_rates_df = pd.DataFrame()
        for key in pooled_key:
            average_rates = data_dict[key][session].mean()

            average_rates_df = pd.concat([average_rates_df, average_rates], axis=1)

        pooled_rates_mean[sessions[session]] = average_rates_df.mean(axis=1)
        pooled_rates_sem[sessions[session]] = average_rates_df.sem(axis=1)

    return pooled_rates_mean, pooled_rates_sem


def get_pooled_avg_corrs(
    raw_rates_dict,
    placecell_rates_dict,
    sort_by_index,
    compare_to_place,
    pooled_key,
    opto_bins,
    control_bins,
):

    corr_diags_mat = [[], [], []]
    opto_corr_zones = [[], [], []]
    control_corr_zones = [[], [], []]
    for key in pooled_key:
        placecell_rates = placecell_rates_dict[key]
        raw_rates = raw_rates_dict[key]
        # sort the ratemaps based on the user input
        compare_rates = placecell_rates[sort_by_index]
        sort_id, _ = sort_ratemaps(compare_rates)
        # using the indices of sort_id Sort each placecell and rawrates  DataFrames in the list of dataframes and s and convert
        place_rates_sorted = [df.reindex(sort_id).fillna(0) for df in placecell_rates]
        raw_rates_sorted = [df.reindex(sort_id).fillna(0) for df in raw_rates]

        # Proccess which rate maps to display
        if compare_to_place:
            display_rates = place_rates_sorted
        else:
            display_rates = raw_rates_sorted.copy()
            display_rates[sort_by_index] = place_rates_sorted[sort_by_index]

        # create the correlation matrices
        array1 = display_rates[sort_by_index].values
        array2 = [df.values for df in display_rates]

        for i in range(len(array2)):
            corr_matrix = create_corr_matrix(array1, array2[i])
            corr_matrix_diag = np.diagonal(corr_matrix)
            corr_diags_mat[i].append(corr_matrix_diag)
            opto_corr_mat = np.mean(
                corr_matrix[
                    opto_bins[0] : opto_bins[1] + 1, opto_bins[0] : opto_bins[1] + 1
                ]
            )
            opto_corr_zones[i].append(opto_corr_mat)

            control_corr_mat = np.mean(
                corr_matrix[
                    control_bins[0] : control_bins[1] + 1,
                    control_bins[0] : control_bins[1] + 1,
                ]
            )
            control_corr_zones[i].append(control_corr_mat)

    coor_diags_mean = []
    coor_diags_sem = []
    opto_zone_diag = []
    cont_zone_diag = []

    for session_cor in corr_diags_mat:
        coor_diags_mean.append(np.nanmean(session_cor, axis=0))
        coor_diags_sem.append(stats.sem(session_cor, axis=0))

        opto_diag = []
        cont_diag = []
        for id in session_cor:
            opto_diag.append(np.nanmean(id[opto_bins[0] : opto_bins[1] + 1]))
            cont_diag.append(np.nanmean(id[control_bins[0] : control_bins[1] + 1]))

        opto_zone_diag.append(opto_diag)
        cont_zone_diag.append(cont_diag)

    return (
        coor_diags_mean,
        coor_diags_sem,
        opto_zone_diag,
        cont_zone_diag,
        opto_corr_zones,
        control_corr_zones,
    )


def get_average_rates(data_dict, key):
    average_rates = pd.DataFrame()
    sessions = ["session_1", "session_2", "session_3"]
    for session in range(len(data_dict[key])):

        average_rates[sessions[session]] = data_dict[key][session].mean()

    return average_rates


def create_corr_matrix(array1, array2):
    # takes data in form of np.array matrixes must be same dimention
    corr_matrix = np.empty((array1.shape[1], array2.shape[1]))

    for i in range(array1.shape[1]):
        for j in range(array2.shape[1]):
            corr_matrix[i, j] = np.corrcoef(array1[:, i], array2[:, j])[0, 1]

    return corr_matrix


def spatial_corr(array1, array2):
    spatial_correlations = []
    for i in range(array1.shape[0]):
        spatial_correlations.append(np.corrcoef(array1[i], array2[i])[0, 1])
    return spatial_correlations


# Function to count adjacent bins above threshold
def find_placecell_width(data, threshold):
    '''
    data: a single ratemap containing bins of activity
    threshold: value to class as still being part of place field

    example place cell center = 1 (normalised rate)
    wants to find how many consecutive bins around the center are at or above threshold value. 
   
   '''

    # find the index of the max element(PC center)
    index = np.argmax(data)
    # check there is actually a place field
    if max(data) >= threshold:
        count = 1
    else:
        return 0

    current_index = index

    # Check bins to the right and count if value is above or equal to threshold
    while current_index < len(data) - 1 and data[current_index + 1] >= threshold:
        count += 1
        current_index += 1

    # Check bins to the Left and count if value is above or equal to threshold
    current_index = index
    while current_index > 0 and data[current_index - 1] >= threshold:
        count += 1
        current_index -= 1

    return count

def get_significance_stars(p_val):
    if p_val >0.05:
        sig = "ns"
    elif p_val >0.01:
        sig = "*"
    elif p_val >0.001:
        sig = "**"
    elif p_val >0.0001:
        sig = "***"
    else:
        sig ="****"
    return sig


def get_lap_transitions(track_coordinates, threshold, direction='top'):
    """
    Find the transition points in track coordinates where the values cross a specified threshold.
    
    Parameters:
    - track_coordinates: numpy array of shape (N,) representing the 1D coordinates.
    - threshold: float, the value to use for detecting transitions.
    - direction: str, 'top' for detecting transitions above the threshold, 'bottom' for below.

    Returns:
    - mid_points: numpy array of transition midpoints.
    """
    track_mid_point = 70
    track_reward_zone = track_coordinates.copy()

    

    # Apply thresholding to create a mask based on the direction
    if direction == 'top':
        
        if track_reward_zone[0] > threshold:
            track_reward_zone[0] = track_mid_point

        if track_reward_zone[-1] > threshold:
            track_reward_zone[-1] = track_mid_point

        # Set values below the threshold to 0
        track_reward_zone[uniform_filter1d(track_coordinates, size=20) < threshold] = 0
        # Find transition points based on the change in values
        transition_start = np.where(np.diff(track_reward_zone, axis=0) > 20)[0]
        transition_end = np.where(np.diff(track_reward_zone, axis=0) < -20)[0]

    else:
        
        if track_reward_zone[0] < threshold:
            track_reward_zone[0] = track_mid_point

        if track_reward_zone[-1] < threshold:
            track_reward_zone[-1] = track_mid_point
        
        # Set values above the threshold to 100
        track_reward_zone[uniform_filter1d(track_coordinates, size=20) > threshold] = 100
        transition_end = np.where(np.diff(track_reward_zone, axis=0) > 20)[0]
        transition_start = np.where(np.diff(track_reward_zone, axis=0) < -20)[0]

    try:
        # Compute the midpoints of lap transition
        lap_mid_point = transition_start + (transition_end - transition_start) / 2
        
    except:
        
        print("error computing lap_mid_point")
        # # Handle cases where transitions are not found by adjusting the start and end points
        # if direction == 'top':
            
        #     if track_coordinates[np.where(~np.isnan(track_coordinates))[0][-1]] > threshold:
        #         transition_end = np.append(transition_end, len(track_coordinates))
        #     if track_coordinates[np.where(~np.isnan(track_coordinates))[0][0]] > threshold:
        #         transition_start = np.insert(transition_start, 0, 0)
        #     else: 
                
        #         if len(transition_start)<len(transition_end):
        #             transition_start = np.insert(transition_start, 0, 0)
        #         elif len(transition_start)>len(transition_end):
        #             transition_end = np.append(transition_end, len(track_coordinates))

            
        # else:
            
        #     if track_coordinates[np.where(~np.isnan(track_coordinates))[0][-1]] < threshold:
        #         transition_end = np.append(transition_end, len(track_coordinates))
        #     if track_coordinates[np.where(~np.isnan(track_coordinates))[0][0]] < threshold:
        #         transition_start = np.insert(transition_start, 0, 0)
        #     else: 
        #         # bit of a bodge not sure if this will catch all edge cases
                
        #         if len(transition_start)<len(transition_end):
        #             transition_start = np.insert(transition_start, 0, 0)
        #         elif len(transition_start)>len(transition_end):
        #             transition_end = np.append(transition_end, len(track_coordinates))  
        
        # # Recompute the midpoints of lap transition
        # lap_mid_point = transition_start + (transition_end - transition_start) / 2


    return lap_mid_point



def get_laps_from_coords(track_coordinates, plot_output=False, output_half_laps=False, top_threshold = 110, bottom_threshold = 20):
    """
    Identify lap transitions from 1D coordinates and optionally plot the results.
    
    Parameters:
    - track_coordinates: numpy array of shape (N,) representing the 1D coordinates.
    - plot_output: boolean, if True, plot the coordinates and lap transitions.
    - output_half_laps: boolean, if True, calculate and return interleaved lap lengths.

    Returns:
    - lap_output: numpy array of lap transition points.
    """
    
    # Get transition midpoints for top and bottom thresholds
    top_lap_mid_point = get_lap_transitions(track_coordinates, threshold=top_threshold, direction='top')
    bot_lap_mid_point = get_lap_transitions(track_coordinates, threshold=bottom_threshold, direction='bottom')

    # Determine which set of transitions (top or bottom) should be processed first
    if bot_lap_mid_point[0] < top_lap_mid_point[0]:
        first_laps, second_laps = bot_lap_mid_point, top_lap_mid_point
    else:
        first_laps, second_laps = top_lap_mid_point, bot_lap_mid_point

    # Initialize lap transitions with the first point from the first laps
    lap_transitions = [first_laps[0]]

    # Identify transitions between laps
    for i in range(len(first_laps) - 1):
        # Check if there's an overlap with the second laps within the range of the current first lap
        if any((second_laps >= first_laps[i]) & (second_laps <= first_laps[i + 1])):
            lap_transitions.append(first_laps[i + 1])

    # Convert transitions to integers and round
    lap_transitions = np.round(lap_transitions).astype(int)
    lap_output = lap_transitions

    # Calculate and interleave lap lengths if output_half_laps is True
    if output_half_laps:
        second_lap_transitions = [second_laps[0]]
        for i in range(len(second_laps) - 1):
            if any((first_laps >= second_laps[i]) & (first_laps <= second_laps[i + 1])):
                second_lap_transitions.append(second_laps[i + 1])

        second_lap_transitions = np.round(second_lap_transitions).astype(int)
        min_len = min(len(lap_transitions), len(second_lap_transitions))
        lap_half_transitions = np.zeros(min_len * 2, dtype=int)
        lap_half_transitions[0::2] = lap_transitions[:min_len]
        lap_half_transitions[1::2] = second_lap_transitions[:min_len]

        # Append any remaining transitions if lengths are different
        if len(lap_transitions) > len(second_lap_transitions):
            lap_half_transitions = np.append(lap_half_transitions, lap_transitions[min_len:])
        elif len(second_lap_transitions) > len(lap_transitions):
            lap_half_transitions = np.append(lap_half_transitions, second_lap_transitions[min_len:])
        
        lap_output = lap_half_transitions

    # Plot results if plot_output is True
    if plot_output:
        plt.figure(figsize=(18, 4))
        plt.plot(track_coordinates)
        for xc in lap_output:
            plt.axvline(x=xc, color='g')
        plt.title('Track Coordinates and Lap Transitions')
        plt.xlabel('Index')
        plt.ylabel('Coordinate Value')
        plt.show()

    return lap_output


def hdf5_to_dict(group):
    """
    Recursively convert an HDF5 group to a nested dictionary.
    """
    data_dict = {}
    
    # Iterate over the items in the group
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            # Recursively handle sub-groups
            data_dict[key] = hdf5_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            # Convert datasets to numpy arrays
            data_dict[key] = item[()]
    
    return data_dict


def get_lap_transitions(track_coordinates, threshold, filter_size, direction='top'):
    """
    Find the transition points in track coordinates where the values cross a specified threshold.
    
    Parameters:
    - track_coordinates: numpy array of shape (N,) representing the 1D coordinates.
    - threshold: float, the value to use for detecting transitions.
    - direction: str, 'top' for detecting transitions above the threshold, 'bottom' for below.

    Returns:
    - mid_points: numpy array of transition midpoints.
    """
    
    track_reward_zone = track_coordinates.copy()

    # Apply thresholding to create a mask based on the direction
    if direction == 'top':
        # Set values below the threshold to 0
        track_reward_zone[uniform_filter1d(track_coordinates, size=filter_size) < threshold] = -100
        # Find transition points based on the change in values
        transition_start = np.where(np.diff(track_reward_zone, axis=0) > 20)[0]
        transition_end = np.where(np.diff(track_reward_zone, axis=0) < -20)[0]

        if transition_end[0] < transition_start[0]:
            transition_start = np.insert(transition_start, 0, 0)
        
        if transition_end[-1] < transition_start[-1]:
            transition_end = np.append(transition_end, len(track_coordinates))

    else:
        # Set values above the threshold to 100
        track_reward_zone[uniform_filter1d(track_coordinates, size=filter_size) > threshold] = 300
        transition_end = np.where(np.diff(track_reward_zone, axis=0) > 20)[0]
        transition_start = np.where(np.diff(track_reward_zone, axis=0) < -20)[0]

        if transition_end[0] < transition_start[0]:
            transition_start = np.insert(transition_start, 0, 0)
        
        if transition_end[-1] < transition_start[-1]:
            transition_end = np.append(transition_end, len(track_coordinates))
        

    try:
        # Compute the midpoints of lap transition
        lap_mid_point = transition_start + (transition_end - transition_start) / 2

    except:
        
        # Handle cases where transitions are not found by adjusting the start and end points
        if direction == 'top':
            
            if track_coordinates[np.where(~np.isnan(track_coordinates))[0][-1]] > threshold:
                transition_end = np.append(transition_end, len(track_coordinates))
            if track_coordinates[np.where(~np.isnan(track_coordinates))[0][0]] > threshold:
                transition_start = np.insert(transition_start, 0, 0)
            else: 
                
                if len(transition_start)<len(transition_end):
                    transition_start = np.insert(transition_start, 0, 0)
                elif len(transition_start)>len(transition_end):
                    transition_end = np.append(transition_end, len(track_coordinates))

        else:
            
            if track_coordinates[np.where(~np.isnan(track_coordinates))[0][-1]] < threshold:
                transition_end = np.append(transition_end, len(track_coordinates))
            if track_coordinates[np.where(~np.isnan(track_coordinates))[0][0]] < threshold:
                transition_start = np.insert(transition_start, 0, 0)
            else: 
                # bit of a bodge not sure if this will catch all edge cases
                
                if len(transition_start)<len(transition_end):
                    transition_start = np.insert(transition_start, 0, 0)
                elif len(transition_start)>len(transition_end):
                    transition_end = np.append(transition_end, len(track_coordinates))  
        
        # Recompute the midpoints of lap transition
        lap_mid_point = transition_start + (transition_end - transition_start) / 2


    return lap_mid_point


def get_laps_from_coords(track_coordinates, plot_output=False, output_half_laps=False, filter_size=20):
    """
    Identify lap transitions from 1D coordinates and optionally plot the results.
    
    Parameters:
    - track_coordinates: numpy array of shape (N,) representing the 1D coordinates.
    - plot_output: boolean, if True, plot the coordinates and lap transitions.
    - output_half_laps: boolean, if True, calculate and return interleaved lap lengths.

    Returns:
    - lap_output: numpy array of lap transition points.
    """
    
    # Define thresholds for detecting laps essentially coords or reward zone
    top_threshold = 100
    bottom_threshold = 20

    # Get transition midpoints for top and bottom thresholds
    top_lap_mid_point = get_lap_transitions(track_coordinates, threshold=top_threshold, direction='top', filter_size=filter_size)
    bot_lap_mid_point = get_lap_transitions(track_coordinates, threshold=bottom_threshold, direction='bottom', filter_size=filter_size)


    # Determine which set of transitions (top or bottom) should be processed first
    if bot_lap_mid_point[0] < top_lap_mid_point[0]:
        first_laps, second_laps = bot_lap_mid_point, top_lap_mid_point
    else:
        first_laps, second_laps = top_lap_mid_point, bot_lap_mid_point

    # Initialize lap transitions with the first point from the first laps
    lap_transitions = [first_laps[0]]

    # Identify transitions between laps
    for i in range(len(first_laps) - 1):
        # Check if there's an overlap with the second laps within the range of the current first lap
        if any((second_laps >= first_laps[i]) & (second_laps <= first_laps[i + 1])):
            lap_transitions.append(first_laps[i + 1])

    # Convert transitions to integers and round
    lap_transitions = np.round(lap_transitions).astype(int)
    lap_output = lap_transitions

    # Calculate and interleave lap lengths if output_half_laps is True
    if output_half_laps:
        second_lap_transitions = [second_laps[0]]
        for i in range(len(second_laps) - 1):
            if any((first_laps >= second_laps[i]) & (first_laps <= second_laps[i + 1])):
                second_lap_transitions.append(second_laps[i + 1])

        second_lap_transitions = np.round(second_lap_transitions).astype(int)
        min_len = min(len(lap_transitions), len(second_lap_transitions))
        lap_half_transitions = np.zeros(min_len * 2, dtype=int)
        lap_half_transitions[0::2] = lap_transitions[:min_len]
        lap_half_transitions[1::2] = second_lap_transitions[:min_len]

        # Append any remaining transitions if lengths are different
        if len(lap_transitions) > len(second_lap_transitions):
            lap_half_transitions = np.append(lap_half_transitions, lap_transitions[min_len:])
        elif len(second_lap_transitions) > len(lap_transitions):
            lap_half_transitions = np.append(lap_half_transitions, second_lap_transitions[min_len:])
        
        lap_output = lap_half_transitions

    # Plot results if plot_output is True
    if plot_output:
        plt.figure(figsize=(18, 4))
        plt.plot(track_coordinates)
        for xc in lap_output:
            plt.axvline(x=xc, color='g')
        plt.title('Track Coordinates and Lap Transitions')
        plt.xlabel('Index')
        plt.ylabel('Coordinate Value')
        plt.show()

    return lap_output



def get_even_odd_ratemaps(coordinates,times,event_time_dictionary):

    coords = coordinates
    number_bins = 40
    full_track_length = 141
    reward_length = 16
    velocity_threshold = 7
    sigma = 1.5
    track_length = full_track_length - (2*reward_length)
    time_per_frame = times.max() / times.shape[0]
    
    # calculate the velocity
    velocity = abs(np.diff(coords))/np.diff(times)
    velocity = uniform_filter1d(velocity,60)
    velocity = np.append(velocity,velocity[-1])

    # get the odd laps 
    laps_data = get_laps_from_coords(coords, plot_output=False, filter_size=50, output_half_laps=False)
    
    # start and end laps from 0 and end of data to use all data 
    laps_data = np.insert(laps_data, 0, 0)
    laps_data = np.append(laps_data, len(coords))

    # create empty array of zeros
    odd_laps = np.zeros_like(coords)
    index_range = len(laps_data) if len(laps_data)%2 == 0 else len(laps_data)-1
    for i in range(index_range):
        # for even idices (odd laps) fill with 1's between indeces
        if i%2 == 0:
            odd_laps[laps_data[i]:laps_data[i+1]] = 1
    odd_laps = odd_laps.astype(bool)
    even_laps = ~odd_laps

    # calculate the bin edges for the ratemaps
    bin_size = track_length/ number_bins
    bin_edges = np.zeros((1,number_bins+1))[0]
    bin_edges[0] = reward_length
    for i in range(1, number_bins+1):
        bin_edges[i] = bin_edges[i-1]+bin_size

    # initialise loop through all cells in events dictionary
    odd_lap_ratemaps = []
    even_lap_ratemaps = []
    cell_ids = []
    for cell_id, event_times in event_time_dictionary.items():
        
        event_frames=[]
        oddlap_event = []
        evenlap_event = []
        event_coords = []
        event_vel = []
        for event in event_times:
            frame = abs(times-event).argmin()
            event_frames.append(frame)
            event_coords.append(coords[frame])
            event_vel.append(velocity[frame])

            oddlap_event.append(odd_laps[frame])
            evenlap_event.append(even_laps[frame])

        event_coords = np.array(event_coords)

        oddlap_event = np.array(oddlap_event).astype(bool)
        evenlap_event = np.array(evenlap_event).astype(bool)
        event_vel = np.array(event_vel)

        # discard events that are out of the track limits ie in reward zones
        out_of_track_coords = (event_coords<bin_edges[0]) | (event_coords> bin_edges[-1])
        below_velocity_thresh = (event_vel<velocity_threshold)
        event_masks = out_of_track_coords*below_velocity_thresh

        event_coords_track = event_coords[~event_masks]
        oddlap_event_track = oddlap_event[~event_masks]
        evenlap_event_track = evenlap_event[~event_masks]

        # filter the events for even and odd laps
        odd_lap_events = event_coords_track[oddlap_event_track]
        even_lap_events = event_coords_track[evenlap_event_track]

        # find the number of events per bin using histogram
        events_per_bin_odd, edges = np.histogram(
            odd_lap_events,
            bins=bin_edges)
        events_per_bin_even, edges = np.histogram(
            even_lap_events,
            bins=bin_edges)

        
        # calculate the time spent in each bin (for odd and even laps)

        # filter data for odd and even and above velocity 
        active_odd_laps = np.logical_and(odd_laps, velocity>velocity_threshold)
        active_even_laps = np.logical_and(even_laps, velocity>velocity_threshold)

        # coords of odd and even laps
        odd_lap_coords = coords[active_odd_laps]
        even_lap_coords = coords[active_even_laps]

        # number of odd frames per bin
        frames_per_bin_odd, edges = np.histogram(
            odd_lap_coords,
            bins=bin_edges)

        # number of even frames per bin
        frames_per_bin_even, edges = np.histogram(
            even_lap_coords,
            bins=bin_edges)

        #calculate the time sepent in each bin in seconds
        time_per_bin_odd = frames_per_bin_odd * time_per_frame
        time_per_bin_even = frames_per_bin_even * time_per_frame


        # calculate the smoothed ratemaps

        #smooth the events and times 
        events_per_bin_odd_gauss = gaussian_filter(events_per_bin_odd.astype(float),sigma=sigma,truncate=2)
        time_per_bin_odd_gauss = gaussian_filter(time_per_bin_odd,sigma=sigma,truncate=2)
        #calculate event rate and normalise
        smoothed_eventrate_odd_laps = events_per_bin_odd_gauss/time_per_bin_odd_gauss
        norm_smoothed_eventrate_odd_laps = smoothed_eventrate_odd_laps/(max(smoothed_eventrate_odd_laps) + 1e-10)

        #smooth the events and times 
        events_per_bin_even_gauss = gaussian_filter(events_per_bin_even.astype(float),sigma=sigma,truncate=2)
        time_per_bin_even_gauss = gaussian_filter(time_per_bin_even,sigma=sigma,truncate=2)
        #calculate event rate and normalise
        smoothed_eventrate_even_laps = events_per_bin_even_gauss/time_per_bin_even_gauss
        norm_smoothed_eventrate_even_laps = smoothed_eventrate_even_laps/(max(smoothed_eventrate_even_laps) + 1e-10)

        # append each cells ratemaps
        even_lap_ratemaps.append(norm_smoothed_eventrate_even_laps)
        odd_lap_ratemaps.append(norm_smoothed_eventrate_odd_laps)

        #append the cell ids
        cell_ids.append(cell_id)
    
    return even_lap_ratemaps, odd_lap_ratemaps, cell_ids


def sort_ratemaps(rates):

    maxrate_index = np.zeros(rates.shape[0], dtype=int)
    for i in range(rates.shape[0]):
        idx = np.argmax(rates.iloc[i, :])
        maxrate_index[i] = idx

    # Sort indices in ascending order
    id = np.argsort(maxrate_index)[::-1]
    idexnames = rates.index[id]

    return idexnames, maxrate_index


def interp_for_nans(coords):
    # Indices where values are not NaN
    not_nan_indices = np.where(~np.isnan(coords))[0]

    # Indices where values are NaN
    nan_indices = np.where(np.isnan(coords))[0]

    # Interpolating NaN values
    coords[nan_indices] = np.interp(nan_indices, not_nan_indices, coords[not_nan_indices])

    return coords
