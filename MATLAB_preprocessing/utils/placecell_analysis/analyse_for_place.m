%% binning the data
% Will eventually need to code for the left and right direction based on
% the LEDs currently have done this manually.

%% analysis proper


function [place_cells, rate_maps] = analyse_for_place(track_data,cell_data,varargin)


P = inputParser;
P.addParameter('number_bins',25);
P.addParameter('track_length',141);
P.addParameter('reward_zone_len',16);
P.addParameter('eventrate_thresh',5);
P.addParameter('analyse_reward_zone',true,@islogical);
P.addParameter('reward_zone_bodge',0);
P.addParameter('perturb_bin',false,@islogical);
P.addParameter('bin_range_perturb',[10,18]);
P.addParameter('perturb_scaler',0.3);


P.parse(varargin{:});
for i=fields(P.Results)'
   eval([i{1} '=P.Results.(i{1});']); 
end



% number_bins = 25; % number of bins to divide track by 40 is around 3.5cm 
% track that im using is 141cm with 7cm reward zones
%select the events to be analysed 
%Data_set = tracked_data.session7;




% set up the bin coordinates
%x_pixsize = size(Data_set.Track_image,2); % get the total number pixels in x dim
% x_dimentions = Track_length;
% x_cords = (14:x_dimentions); %create and array of values for each pixel
% [~, edges] = discretize(x_cords,number_bins); %create bined data evenly distribute bins across the track.
% %%

%%to get the edges minus the reward area

if analyse_reward_zone %if you want to include reward zone in analysis
   reward_zone_len = 0;
   reward_zone_bodge = 0;
end



reward_zone_start = reward_zone_len;
reward_zone_end = (reward_zone_len - reward_zone_bodge);



track_len_to_analyse = track_length-(reward_zone_start+reward_zone_end);
bin_size = track_len_to_analyse/number_bins;

bin_edges = zeros(1,number_bins+1);
bin_edges(1) = reward_zone_len; 

for e = 2:number_bins+1

    binedge = bin_edges(e-1);  
    bin_edges(e) = binedge+bin_size;
end


%% Loop through both the right and the left placecells

%get vars from the track_data
track_cordinates = track_data.track_cordinates;
frame_time = track_data.frame_time;


left_runs = track_data.left_runs;
right_runs = track_data.right_runs;

left_frames = find(~isnan(left_runs));
right_frames = find(~isnan(right_runs));



time_per_frame = max(frame_time)/size(track_cordinates,1);


Traversals = ["right", "left"];

% create some empty arrrays
place_cells = struct;
place_cells.left = struct;
place_cells.right = struct;
all_Place_cells = struct;
rate_maps.left_PC_eventrates = [];
rate_maps.right_PC_eventrates = [];
rate_maps.left_all_eventrates = [];
rate_maps.right_all_eventrates =[];

rate_maps.right_all_random_shuf = [];
rate_maps.left_all_random_shuf = [];
rate_maps.right_PC_random_shuf = [];
rate_maps.left_PC_random_shuf =[];

All_cell_ids = string(fieldnames(cell_data.all_cells));

for c = 1:(size(All_cell_ids,1))

    cell_event_amps = cell_data.all_cells.(All_cell_ids(c)).amps;
    event_x_coords = cell_data.all_cells.(All_cell_ids(c)).x_coord;
    right_run = cell_data.all_cells.(All_cell_ids(c)).right_run;
    left_run = cell_data.all_cells.(All_cell_ids(c)).left_run;
    
    %  This is where we need to randomly delete 30% of spikes but only in optozone 
    if perturb_bin
        
        % a mask for each events with of random 1s at a chance level of perturb_scaler
        chance_mask = rand(size(event_x_coords))< perturb_scaler;
        
        % a mask for if the events is in the perturb area of track
        perturb_start = bin_edges(bin_range_perturb(1));
        perturb_end = bin_edges(bin_range_perturb(2))+bin_size;
        perturb_zone_mask = event_x_coords>=perturb_start & event_x_coords<=perturb_end;
        
        % mask to delete events at chance of perturb_scaler, only if
        % within perturb zone
        perturb_mask = logical(perturb_zone_mask.*chance_mask);
    
        % delete events outside track limits
        cell_event_amps(perturb_mask)=[];
        event_x_coords(perturb_mask)=[];
        left_run(perturb_mask)=[];
        right_run(perturb_mask)=[];
    
        
        cell_data.all_cells.(All_cell_ids(c)).amps = cell_event_amps;
        cell_data.all_cells.(All_cell_ids(c)).x_coord = event_x_coords;
        cell_data.all_cells.(All_cell_ids(c)).left_run = left_run;
        cell_data.all_cells.(All_cell_ids(c)).right_run = right_run;
        
         % might not be a right cell
        if isfield(cell_data.right_cells, All_cell_ids(c))
            cell_data.right_cells.(All_cell_ids(c)).amps = cell_event_amps(logical(right_run));
            cell_data.right_cells.(All_cell_ids(c)).x_coord = event_x_coords(logical(right_run));
            cell_data.right_cells.(All_cell_ids(c)).left_run = left_run(logical(right_run));
            cell_data.right_cells.(All_cell_ids(c)).right_run = right_run(logical(right_run));
        end
        
        
    
        if isfield(cell_data.left_cells, All_cell_ids(c)) % might not be a left cell
            cell_data.left_cells.(All_cell_ids(c)).amps = cell_event_amps(logical(left_run));
            cell_data.left_cells.(All_cell_ids(c)).x_coord = event_x_coords(logical(left_run));
            cell_data.left_cells.(All_cell_ids(c)).left_run = left_run(logical(left_run));
            cell_data.left_cells.(All_cell_ids(c)).right_run = right_run(logical(left_run));
        end
        
    end
           
end


%loop through both the right and then left traversals of the track
for runs = 1:size(Traversals,2)

    Place_direction = Traversals(runs);
    
    try
      event_struct = cell_data.(strcat('events_',Place_direction)); %get the cell by cell event data for either left or right
    catch
      event_struct = cell_data.(strcat(Place_direction,'_cells'));
    end
      cells = string(fieldnames(event_struct));
    
     
    %% loop through each set of cells and analyse for place.
    
    h = waitbar(0,'Please wait...','Name',strcat('Analysing'," ",Place_direction,' Placecells'));
    disp(strcat('Analysing'," ",Place_direction,' placecells ...please wait'));
    for c =1:size(cells,1)
        
            
        try    
            cell_events = event_struct.(cells(c)).frame_number; %get the frame number of events
            cell_event_amps = event_struct.(cells(c)).Amplitude; %get the amplitude for cell events
            event_x_coords = event_struct.(cells(c)).coordinates(:,1); % get the x track cordinate for each event
        catch
            %tempbodge
            cell_event_amps = event_struct.(cells(c)).amps;
            event_x_coords = event_struct.(cells(c)).x_coord;
            cell_events = event_x_coords; %get the frame number of events
        end 
        
        out_of_track = event_x_coords<bin_edges(1)|event_x_coords>bin_edges(end);% logic of any events that are outside of the Track limits
        
        cell_events(out_of_track)=[]; % get rid of events outside track limits
        cell_event_amps(out_of_track)=[];
        event_x_coords(out_of_track)=[];
        
        %if the number of events is less than 5 dont analyse this cell
        if size(cell_events,1) < eventrate_thresh 
            continue
        end
        
        
        %for each cell will need to look at events per bin
        
        %bodge as some of cordinates are outside bins
        %%%----------------------------------
        % event_x_coords(event_x_coords<0)=0;
        % event_x_coords(event_x_coords>140)=140;
        %%%----------------------------------
        
        
        [event_per_bin, ~, whichbin] = histcounts(event_x_coords,bin_edges); 
        event_per_bin = event_per_bin.';


        binAmps = zeros(number_bins,size(cell_events,1)); %this creates and array and puts the amp of each event into the correct bin
        Amp_per_bin= [];
        for i = 1:size(cell_events,1)
            
            binto = whichbin(i);
            binAmps(binto,i) = cell_event_amps(i);
        
        end
        Amp_per_bin = sum(binAmps,2);
        
        
        
        %%
        
        frames_per_bin = histcounts(track_cordinates(:,1),bin_edges).'; %% should check that we want running time in each bin and not just overal time
        time_per_bin = frames_per_bin*time_per_frame; 
        
        event_rate_per_bin = event_per_bin./time_per_bin;
  
        %smoothed time per bin based on velocity
        
        if Place_direction == "left"
            
            active_frames_per_bin = histcounts(left_runs,bin_edges).'; %% should chaeck that we want running time in each bin and not just overal time
            Frames_directional = left_frames;
        end
        
        if Place_direction == "right"
            active_frames_per_bin = histcounts(right_runs,bin_edges).'; %% should chaeck that we want running time in each bin and not just overal time
            Frames_directional = right_frames;
        end
        
        
        active_time_per_bin = active_frames_per_bin*time_per_frame;
        
        
        Eventrate_active = event_per_bin./active_time_per_bin;
        Eventrate_active(isnan(Eventrate_active))=0; % if there is a NaN make zero (if there is no location data for a certain bin)
        Eventrate_norm = Eventrate_active./max(Eventrate_active);
        
        
        Amp_event_rate_active = Amp_per_bin./active_time_per_bin;
        Amp_event_rate_active(isnan(Amp_event_rate_active))=0; % if there is a NaN make zero (if there is no location data for a certain bin)
        Amp_event_rate_active_norm = Amp_event_rate_active./max(Amp_event_rate_active);
        
        
        
        % Eventrate_cropped = Eventrate_active(2:24,:);
        % Eventrate_cropped_norm = Eventrate_cropped./max(Eventrate_cropped);
        
        %gauss filter for both
        smoothed_bintime = imgaussfilt(active_time_per_bin,1.5);
        smoothed_event_per_bins = imgaussfilt(event_per_bin,1.5);
        smoothed_amp_bins = imgaussfilt(Amp_per_bin,1.5);
        
        
        
        % %************
        % smoothed_bintime = imgaussfilt(active_time_per_bin(2:24,:),1.5,'FilterSize',5);
        % smoothed_event_bins = imgaussfilt(event_per_bin(2:24,:),1.5,'FilterSize',5);
        % %************
        smoothed_event_rate_per_bin= [];
        smoothed_event_rate_per_bin = smoothed_event_per_bins./smoothed_bintime;
        smoothed_event_rate_per_bin(isnan(smoothed_event_rate_per_bin))=0; % if there is a NaN make zero (if there is no location data for a certain bin)
        smoothed_event_rate_per_bin_norm = smoothed_event_rate_per_bin./max(smoothed_event_rate_per_bin);
        
        
        smoothed_amprate= [];
        smoothed_amprate = smoothed_amp_bins./smoothed_bintime;
        smoothed_amprate(isnan(smoothed_amprate))=0; % if there is a NaN make zero (if there is no location data for a certain bin)
        smoothed_amptrate_norm = smoothed_amprate./max(smoothed_amprate);
        
        
        smoothed_amp_bins_norm = smoothed_amp_bins./max(smoothed_amp_bins);
        
        % figure
        % plot(Eventrate_norm)
        % hold on 
        % plot(smoothed_eventrate_norm)
        
        
        % bins = linspace(1,number_bins,number_bins);
        % f = gaussfilt(bins,Eventrate_left_smoothed,2);
        % figure
        % plot(Eventrate_left_smoothed)
        % hold on 
        % plot(f)
        
        %% have the bin count now need normalised rate accross the bins 
        %so do a percentage of max?
        
        %encoding for spatial information. equation from rubin et al 2015 DOI: 10.7554/eLife.12247
        
        
        % %%%%%*******temp****
        % Eventrate_active = Eventrate_cropped;
        % 
        % %%%%%*******temp****
        
        Ev = Eventrate_active;
        Tb = active_time_per_bin;
        
        Eventrate_active_smoothed = imgaussfilt(Eventrate_active,1.5); 
        
        Ev = smoothed_event_rate_per_bin;
        Tb = active_time_per_bin;
        
        rmean = mean(Ev); % mean event rate
        total_time = sum(Tb); %total time per bin
        
        
        %total_time = sum(Time_per_bin);
        
        spaceinfo = double.empty;
        spatial_info = double.empty;
        
        % number_bins = 23;
        
        for i = 1:number_bins    
            r = Ev(i); % event rate in bin
            p = Tb(i)/total_time; % probability of mouse being in bin
            %p = Time_per_bin(i)/total_time;
            
            spaceinfo(i) = (p*(r/rmean))*(log2(r/rmean));
            
        end
        spatial_info = sum(spaceinfo,'omitnan');
        
        %%random suffle 
        %
        % shuffle the location of the events for a particular cell and then get a
        % distribution of the spatial information.
        spatial_rand = zeros(1000,number_bins);
        spaceinfo_rand = [];
        spatial_info_rand = [];
        
        for t = 1:1000
            %random coordinates 
            rand_event_frames = randsample(Frames_directional, size(cell_events,1));
            rand_event_cord = track_cordinates(rand_event_frames,1);
            
            rand_event_per_bin = histcounts(rand_event_cord,bin_edges).';
            
            smoothed_rand_event_per_bin = imgaussfilt(rand_event_per_bin,1.5);
            rand_event_rate_smoothed = smoothed_rand_event_per_bin./smoothed_bintime; 
            
            rand_event_rate = rand_event_per_bin./active_time_per_bin;
            rand_event_rate(isnan(rand_event_rate))=0; %make nans zeros for when there is no location data for set bin
            
            rand_event_rate_filt = imgaussfilt(rand_event_rate,1.5); 
            
            rand_event_rate_filt_norm = rand_event_rate_filt./max(rand_event_rate_filt);
            
            
            % %%%%% ******temp****
            % rand_event_rate = rand_event_rate(2:24,:);
            % %%%%%*******temp****
            Ev = rand_event_rate;
            Tb = active_time_per_bin;
            
            Ev = rand_event_rate_smoothed;
            Tb = active_time_per_bin;
            
            rmean = mean(Ev); % mean event rate
            total_time = sum(Tb); %total time per bin
            
            for k = 1:number_bins    
                r = Ev(k); % event rate in bin
                p = Tb(k)/total_time; % probability of mouse being in bin
                %p = Time_per_bin(k)/total_time; 
                spaceinfo_rand(k) = (p*(r/rmean))*(log2(r/rmean));
            end
            
            spatial_rand(t,:) = spaceinfo_rand;

        end 
        
        spatial_info_rand = sum(spatial_rand,2,'omitnan');
        
        
        
        %% to calculate a p value we first need to find the z value 
        %  we can get the z value from the simulated random population distribution
        %  z = (value - pop_mean / pop_standard_dev) from this we can calculate p
        %  this can be done automatically by normcdf or the ztest function.
        
        popmean = mean(spatial_info_rand);
        popstdv = std(spatial_info_rand);
        
        %p-value from z depends on one tail or two tail. right tail indicates that
        % we are testing to see if cell has more spatial info than random. 
        
        [~,p] = ztest(spatial_info,popmean,popstdv,'Tail','right'); 
        
        all_p_value(c) = p; 
        
        if p <= 0.05
            place_cells.(Place_direction).(cells(c)) = event_struct.(cells(c));
            place_cells.(Place_direction).(cells(c)).Placecell = 'True';
            place_cells.(Place_direction).(cells(c)).Placecell_stats.spatial_info = spatial_info;
            place_cells.(Place_direction).(cells(c)).Placecell_stats.spatial_info_shuffled = spatial_info_rand;
            place_cells.(Place_direction).(cells(c)).Placecell_stats.p_value = p;
            place_cells.(Place_direction).(cells(c)).Placecell_stats.event_rate = Eventrate_active;
            place_cells.(Place_direction).(cells(c)).Placecell_stats.event_rate_norm = Eventrate_norm;
            place_cells.(Place_direction).(cells(c)).Placecell_stats.event_rate_norm_smoothed = smoothed_event_rate_per_bin_norm;
            place_cells.(Place_direction).(cells(c)).Placecell_stats.Amp_event_rate_norm = smoothed_amptrate_norm;
            
            if Place_direction == "left"
                
                rate_maps.left_PC_eventrates = cat(1,rate_maps.left_PC_eventrates,smoothed_event_rate_per_bin_norm.');
                rate_maps.left_PC_random_shuf = cat(1,rate_maps.left_PC_random_shuf,rand_event_rate_filt_norm.');
            else 
                rate_maps.right_PC_eventrates = cat(1,rate_maps.right_PC_eventrates,smoothed_event_rate_per_bin_norm.');
                rate_maps.right_PC_random_shuf = cat(1,rate_maps.right_PC_random_shuf,rand_event_rate_filt_norm.');
            end
        
        
        end
        
        place_cells.all_cells.(Place_direction).(cells(c)).event_rate_norm_smoothed = smoothed_event_rate_per_bin_norm;
        place_cells.all_cells.(Place_direction).(cells(c)).Amp_event_rate_norm = smoothed_amptrate_norm;
        %all_Place_cells.(Place_direction).all_placecells.(cells(cid)).Amp_event_rate_norm =smoothed_amp_bins_norm;
        
        if Place_direction == "left"
            rate_maps.left_all_eventrates = cat(1,rate_maps.left_all_eventrates,smoothed_event_rate_per_bin_norm.');
            rate_maps.left_all_random_shuf = cat(1,rate_maps.left_all_random_shuf,rand_event_rate_filt_norm.');
        else
            rate_maps.right_all_eventrates = cat(1,rate_maps.right_all_eventrates,smoothed_event_rate_per_bin_norm.');
            rate_maps.right_all_random_shuf = cat(1,rate_maps.right_all_random_shuf,rand_event_rate_filt_norm.');
        end
        
        
        h = waitbar(c/size(cells,1),h,...
        ['Remaining Cells =',num2str(c-1),'/',num2str(size(cells,1))]);
        
    end
    
    close(h)

end

%% get the rate maps for all of the cells regardless of the run direction

All_cell_ids = string(fieldnames(cell_data.all_cells));


for cell = 1:(size(All_cell_ids,1))
    
    
    event_x_coords = cell_data.all_cells.(All_cell_ids(cell)).x_coord;

    %filter out events that are not in the analysis track.
    within_track = event_x_coords>bin_edges(1)&event_x_coords<bin_edges(end);% logic of any events that are outside of the Track limits

    event_x_coords(~within_track)=NaN;

    %filter out events that are not part of a run
    right_run = cell_data.all_cells.(All_cell_ids(cell)).right_run;
    left_run = cell_data.all_cells.(All_cell_ids(cell)).left_run;
    within_run = (right_run | left_run);
    event_x_coords(~within_run)=NaN;

    event_x_coords(isnan(event_x_coords)) = []; %remove NaN values

    %calculate the number of events per bin
    [event_per_bin, ~, whichbin] = histcounts(event_x_coords,bin_edges); 
    event_per_bin = event_per_bin.';
    smoothed_event_per_bins = imgaussfilt(event_per_bin,1.5);
    
    %filter for the frames that are part of a run 
    rightruns = ~isnan(track_data.right_runs);
    leftruns = ~isnan(track_data.left_runs);
   
    filtered_track_cords = track_cordinates(rightruns|leftruns,:);
    
    %calculate the time in each bin from frames per bin
    frames_per_bin = histcounts(filtered_track_cords(:,1),bin_edges).';
    time_per_bin = frames_per_bin*time_per_frame;
    smoothed_time_per_bin = imgaussfilt(time_per_bin,1.5);

    
    %calculate eventrate per bin from events and time per bin
    event_rate_per_bin = event_per_bin./time_per_bin;
    event_rate_per_bin(isnan(event_rate_per_bin))=0; % if there is a NaN make zero (if there is no location data for a certain bin)
    
    event_rate_per_bin_norm = event_rate_per_bin./max(event_rate_per_bin);
    event_rate_per_bin_norm(isnan(event_rate_per_bin_norm))=0;
    
    rate_maps.all_cell_ratemaps(cell,:) = event_rate_per_bin;
    rate_maps.all_cell_ratemaps_norm(cell,:) = event_rate_per_bin_norm;


    %calculate smoothed eventrate per bin
    smoothed_event_rate_per_bin = smoothed_event_per_bins./smoothed_time_per_bin;
    smoothed_event_rate_per_bin(isnan(smoothed_event_rate_per_bin))=0; % if there is a NaN make zero (if there is no location data for a certain bin)
    
    smoothed_event_rate_per_bin_norm = smoothed_event_rate_per_bin./max(smoothed_event_rate_per_bin);
    smoothed_event_rate_per_bin_norm(isnan(smoothed_event_rate_per_bin_norm))=0;


    rate_maps.smooth_all_cell_ratemaps(cell,:) = smoothed_event_rate_per_bin;
    rate_maps.smooth_all_cell_ratemaps_norm(cell,:) = smoothed_event_rate_per_bin_norm;


end




%find unique place cells
    %PCs_right = '';
    %PCs_left = '';

    PCs_right = string(fieldnames(place_cells.right));
    PCs_left = string(fieldnames(place_cells.left));
    
    PCs_right_num = size(PCs_right,1);
    PCs_left_num = size(PCs_left,1);
    
    PCs_left_right = cat(1,PCs_right,PCs_left);
    
    % find PCs that only have a left or right PF (ie delete cells that are left&right PCs)
    idx=find(ismember(PCs_right,PCs_left));
    Dual_PCs = PCs_right(idx);
    
    
    Unique_PCs = setdiff(PCs_left_right,Dual_PCs);
    num_dual_PCs = size(Dual_PCs,1);
    num_unique_PCs = size(Unique_PCs,1);
    
    
    unique_ratemaps = [];
    place_cells.PCs_unique = struct;
    cellsleft = place_cells.left;
    cellsright = place_cells.right;
    for c= 1:size(Unique_PCs,1)
        try
            place_cells.PCs_unique.(Unique_PCs(c)) = cellsleft.(Unique_PCs(c));
            unique_ratemaps(c,:) = cellsleft.(Unique_PCs(c)).Placecell_stats.event_rate_norm_smoothed.';
        catch

        end
        
        try
            place_cells.PCs_unique.(Unique_PCs(c)) = cellsright.(Unique_PCs(c));
            unique_ratemaps(c,:) = cellsright.(Unique_PCs(c)).Placecell_stats.event_rate_norm_smoothed.';
        catch
        end
        
    end

    rate_maps.PCs_unique_ratemaps = unique_ratemaps;


%find prefered place cells
    place_cells.PCs_All_prefered = place_cells.PCs_unique;
    PCs_All_prefered_rates = unique_ratemaps;
    
    for d= 1:num_dual_PCs
    dual_left = max(cellsleft.(Dual_PCs(d)).Placecell_stats.event_rate);
    dual_right = max(cellsright.(Dual_PCs(d)).Placecell_stats.event_rate);
        if dual_left>= dual_right
            place_cells.PCs_All_prefered.(Dual_PCs(d)) = cellsleft.(Dual_PCs(d));
            place_cells.PCs_All_prefered.(Dual_PCs(d)).PrefDirection = "Left";
            rates = cellsleft.(Dual_PCs(d)).Placecell_stats.event_rate_norm_smoothed.';
            PCs_All_prefered_rates = [PCs_All_prefered_rates; rates];
        else
            place_cells.PCs_All_prefered.(Dual_PCs(d)) = cellsright.(Dual_PCs(d));
            place_cells.PCs_All_prefered.(Dual_PCs(d)).PrefDirection = "Right";
            rates = cellsright.(Dual_PCs(d)).Placecell_stats.event_rate_norm_smoothed.';
            PCs_All_prefered_rates = [PCs_All_prefered_rates; rates];
        end
    end

    rate_maps.PCs_All_prefered_ratemaps = PCs_All_prefered_rates;

    place_cells.unique_PCs = Unique_PCs;
    place_cells.dual_PCs = Dual_PCs;
    place_cells.all_cells_id = All_cell_ids;
    place_cells.PC_stats.num_dual_PCs = num_dual_PCs;
    place_cells.PC_stats.num_unique_PCs = num_unique_PCs;
    place_cells.PC_stats.num_left_PCs = PCs_left_num;
    place_cells.PC_stats.num_right_PCs = PCs_right_num;

    place_cells.analysis_vars.number_bins = number_bins;
    place_cells.analysis_vars.track_length = track_length;
    place_cells.analysis_vars.reward_zone_len =reward_zone_len;
    place_cells.analysis_vars.reward_zone_bodge = reward_zone_bodge;
    place_cells.analysis_vars.eventrate_thresh = eventrate_thresh;
    place_cells.analysis_vars.analyse_reward_zone = analyse_reward_zone;




end



