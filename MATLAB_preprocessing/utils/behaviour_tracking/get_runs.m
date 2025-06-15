function [left_runs, left_epochs, right_runs, right_epochs] = get_runs(vel_filt_left, vel_filt_right,run_thresh)

%% this function takes the left and right run frames and splits them into epochs (start and finish of run)
% it then filters the the epochs to only keep runs that are longer than the run_thresh

% output for the runs is the track_coordinates for the runs. 
% output for the epochs are the frame number of the run. this can be mapped
% to the time from frame time

for d=1:2 %loop through for left and then right directions
    
    if d ==1
        xLoc = [];
        xLoc  = vel_filt_left(:,1);
    else 
        xLoc = [];
        xLoc  = vel_filt_right(:,1);
    end
    
  
% based on a function ThreshEpochs adpted from Dombeck lab to get epochs

    in = ~isnan(xLoc); % get all frames that are not a nan
    
    ons = find(diff(in)==1); % find the frames of(in) it switches from 0 to 1 (ie when a run starts)
    if in(1), ons = [1;ons]; end % if the first frame is start of run add this as well  
    
    offs = find(diff(in)==-1); % find the frames of(in) where it switches from 1 to 0 (ie when a run ends)
    if in(end), offs = [offs;numel(in)];end % if the last frame is end of run add this as well  
    
    epochs = [(ons+1) offs]; %create a vector of start and stop frames for the runs


    % then find difference in co-ordinates (xLoc) between start and stop of
    epochs2 = [];
    epochs2 = epochs(abs(diff(xLoc(epochs),[],2))>run_thresh,:); %get the epochs where runs are longer than the tresholds
   
    
    good_runs_x = zeros(size(xLoc,1),1); 
    
    for i = 1:size(epochs2,1) % for each epoch fill in the co-ordinates to get co-ordinates of each good run
        start = epochs2(i,1);
        fin = epochs2(i,2);

        good_runs_x(start:fin,:) = xLoc(start:fin);
    end
    
    good_runs_x(good_runs_x ==0)=NaN; %all good runs that are zero change to nans

    if d == 1
        left_runs = good_runs_x;
        left_epochs = epochs2;
    else 
        right_runs = good_runs_x;
        right_epochs = epochs2;
    end
end