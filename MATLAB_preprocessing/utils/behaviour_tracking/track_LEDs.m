function [red_cordinates, green_cordinates, frame_time] = track_LEDs(video,varargin)

%edited on the 25/05/22 to give the option to look for blue rather than green LED coordinates

P = inputParser;
P.addParameter('first_frame',1);
P.addParameter('last_frame',video.NumFrames);
P.addParameter('red_threshold',0.5);
P.addParameter('green_threshold',0.075);
P.addParameter('blue_threshold',0.075);
P.addParameter('track_blue',false,@islogical);

P.addParameter('interpolate',false,@islogical);
P.addParameter('crop_y',false,@islogical);
P.addParameter('crop_x',false,@islogical);
P.addParameter('ycrop',1);
P.addParameter('botycrop',0);
P.addParameter('xcrop',1)
P.addParameter('rigxcrop',0)
P.addParameter('foregroundDetect',false,@islogical)
P.addParameter('dilate',false,@islogical)

P.parse(varargin{:});
for i=fields(P.Results)'
   eval([i{1} '=P.Results.(i{1});']); 
end

if ycrop == 0
    ycrop = 1;
end
if xcrop ==0
    xcrop = 1;
end

%set a model for find a blob from threshold
hblob = vision.BlobAnalysis('AreaOutputPort', true, ... % Set blob analysis handling
                            'CentroidOutputPort', true, ... 
                            'BoundingBoxOutputPort', false', ...
                            'MinimumBlobArea', 1, ...
                            'MaximumBlobArea', 500, ...
                            'MaximumCount', 5);

if foregroundDetect
    foregroundDetector = vision.ForegroundDetector('NumTrainingFrames',10,...
                'InitialVariance',0.01);  
end

frame_array = (first_frame:last_frame);
frames = size(frame_array,2);
coordsR = zeros(frames,2); % preallocate the coordinate matrix
coordsG = zeros(frames,2);

Start_f = read(video,first_frame);
Start_t = video.CurrentTime;
frame_time = zeros(frames,1);
h = waitbar(0,'Please wait...','Name','Tracking the behaviour data');

for i = 1:frames
    
    rgbFrame = read(video,frame_array(i)); % Acquire single frame
    if crop_y
    rgbFrame = rgbFrame(ycrop:end-botycrop,:,:);
    end
    if crop_x
    rgbFrame = rgbFrame(:,xcrop:end-rigxcrop,:);
    end
   

    frame_time(i) = (video.currentTime)-Start_t;
    
    %Red Channel    
    diffFrameR = imsubtract(rgbFrame(:,:,1), rgb2gray(rgbFrame)); % Get red component of the image    
    if foregroundDetect       
        foregroundMask = step(foregroundDetector,im2gray(im2single(rgbFrame)));
        diffFrameR(~foregroundMask)=0;
    end
    if dilate
    se = strel('disk',3);
    diffFrameR = imdilate(diffFrameR,se);
    end
    %diffFrameR = medfilt2(diffFrameR, [3 3]);
    binFrameR = imbinarize(diffFrameR, red_threshold); % Convert the image into binary image with the red objects as white

    [areaRed,centroidR] = step(hblob, binFrameR);
    [~,index] = max(areaRed);  
    centroidR = centroidR(index,:);
    
    
    try
        coordsR(i,:) = centroidR;
        if crop_y
        coordsR(i,2) = coordsR(i,2)+ycrop;
        end
        if crop_x
        coordsR(i,1) = coordsR(i,1)+xcrop;
        end        %catches instance where it either cant find a blob.
    catch 
        coordsR(i,:) = NaN;
    end


%Green or blue channel
    if track_blue
        diffFrameG = imsubtract(rgbFrame(:,:,3), rgb2gray(rgbFrame)); % Get blue component of the image        

        if foregroundDetect             
            foregroundMask = step(foregroundDetector,im2gray(im2single(rgbFrame)));
            diffFrameG(~foregroundMask)=0;
        end
        if dilate
        se = strel('disk',3);
        diffFrameG = imdilate(diffFrameG,se);
        end
        %diffFrameG = medfilt2(diffFrameG, [3 3]);
        binFrameG = imbinarize(diffFrameG, blue_threshold); % Convert the image into binary image with the blue objects as white
    
    else
        diffFrameG = imsubtract(rgbFrame(:,:,2), rgb2gray(rgbFrame)); % Get green component of the image
        
        % Filter out the noise by using median filter
        if foregroundDetect 
            foregroundMask = step(foregroundDetector,im2gray(im2single(rgbFrame)));
            diffFrameG(~foregroundMask)=0;
        end
        if dilate
        se = strel('disk',3);
        diffFrameG = imdilate(diffFrameG,se);
        end
        %diffFrameG = medfilt2(diffFrameG, [3 3]);
        binFrameG = imbinarize(diffFrameG, green_threshold); % Convert the image into binary image with the green objects as white 
    end
    
    [areaGreen,centroidG] = step(hblob, binFrameG);
    [~,index] = max(areaGreen);  
    centroidG = centroidG(index,:);
    % Get the centroids and bounding boxes of the blobs

    try
        coordsG(i,:) = centroidG; % catches instance where it either cant find a blob
        if crop_y
        coordsG(i,2) = coordsG(i,2)+ycrop;
        end  
        if crop_x
        coordsG(i,1) = coordsG(i,1)+xcrop; 
        end
    catch 
        coordsG(i,:) = NaN;
    end
        
    
    if mod(i,50)==1 % every 100th Frame update progress
        h = waitbar(i/frames,h,...
        ['remaining Frames =',num2str(i-1),'/',num2str(frames)]);
    end

end
close(h)

red_cordinates = coordsR;
green_cordinates = coordsG;

if interpolate % if interpolate is set to true then interpolate the coordinates.
   red_cordinates = nan_interp(red_cordinates);
   green_cordinates = nan_interp(green_cordinates);

end 
    
end

function X_interp = nan_interp(X)

for p = 1:size(X,2)    
t = X(:,p);    
idx = ~isnan(t);
X_interp(:,p) = interp1(find(idx),t(idx),(1:numel(t))');
end

end