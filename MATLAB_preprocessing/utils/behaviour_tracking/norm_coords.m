function [coord_norm_cm,coord_norm, conversions] = norm_coords(v,coordinates,varargin)


P = inputParser;
P.addParameter('outputplot',false,@islogical);
P.addParameter('track_length',141);
P.addParameter('track_width',1);
P.addParameter('start_frame',1);
% can provide rotated and croped might be useful for multi batch conversions
P.addParameter('crop',[]); 
P.addParameter('rotation',[]);
P.addParameter('reward_length',14); % size of the reward zone in cm;


P.parse(varargin{:});
for i=fields(P.Results)'
   eval([i{1} '=P.Results.(i{1});']); 
end


if isempty(crop)||isempty(rotation)
   % get the crop and the rotations from the video frame.
[crop, rotation] = video_crop_rotate(v,start_frame); % this is a function below
end

% use the rotate and crop to translate the image

im = read(v,start_frame); 
alpha = rotation; %get the rotation of the image
coordsyx = flip(coordinates,2); 

% creates a rotation matrix for the co-ordinates and then rotates the
% coordinates

RotatedIm = imrotate(im,alpha);   % rotation of the main image (im)
RotMatrix = [cosd(alpha) -sind(alpha); sind(alpha) cosd(alpha)]; 
track_cen = (size(im(:,:,1))/2)';         % Center of the main image
track_rot_cent = round((size(RotatedIm(:,:,1) )/2)');  % Center of the transformed image
Rotated_coords = (RotMatrix*(coordsyx.'-track_cen)+track_rot_cent).';
Rotated_coords = flip(Rotated_coords,2);
 

%% Crop the image and translate the coordinates
CroppedIm =  RotatedIm(crop(1,1):crop(1,2),crop(2,1):crop(2,2),:);
x_crop = crop(2,1); 
y_crop = crop(1,1);

coord_norm(:,1) = Rotated_coords(:,1)-x_crop; % subtract the crop factor to normalise coordinates
coord_norm(:,2) = Rotated_coords(:,2)-y_crop;


%% translate the coordinates into cms

x_size = size(CroppedIm,2);
y_size = size(CroppedIm,1);


track_length = 141; %length of the track in cms (outside to outside)
track_width = 17; % width of running track in cms from edge of reward 

x_scale = track_length/x_size;
y_scale = track_width/y_size;


coord_norm_cm(:,1) = coord_norm(:,1)*x_scale;
coord_norm_cm(:,2) = coord_norm(:,2)*y_scale;




if outputplot

figure('position', [100,400,1500,500])
subplot(2,1,1)
imshow(CroppedIm)
hold on 
plot(coord_norm(:,1),coord_norm(:,2))

subplot(2,1,2)
plot(coord_norm_cm(:,1))
hold on 
line([1, size(coord_norm_cm(:,1),1)],[reward_length reward_length],'Color','red','LineStyle','--')
hold on 
line([1, size(coord_norm_cm(:,1),1)],[(track_length-reward_length) (track_length-reward_length)],'Color','red','LineStyle','--')

xlabel('Frame Number')
ylabel('Track Length (cm)')

end


conversions.track_length = track_length;
conversions.track_width = track_width;
conversions.x_size = x_size;
conversions.y_size= y_size;
conversions.track_rotation = rotation;
conversions.track_crop = crop;
conversions.reward_length = reward_length;
conversions.cropped_track = CroppedIm;




end



%nested functions

function [vid_crop, rotation] = video_crop_rotate(v,start_frame) 

%% find the croping params for video 
im = read(v, start_frame);
fig = figure('Name', 'Rotate and crop video');

hIm = imshow(im);
fig.WindowState = 'maximized';

sz = size(im);
%pos = [(sz(2)/4) + 0.5, (sz(1)/4) + 0.5, sz(2)/2, sz(1)/2];
rec_pos = [10, 10, 100,100];
r = drawrectangle('Rotatable',true,...
    'DrawingArea','unlimited',...
    'Position',rec_pos,...
    'FaceAlpha',0);
r.Label = 'Rotate rectangle to rotate image';
drawline('Position',[0, sz(1)/4.5; sz(2), sz(1)/4.5],'StripeColor','r');


addlistener(r,'MovingROI',@(src,evt) rotateImage(src,evt,hIm,im));
ROI1 = customWait(r);
close(fig);
rotation =  ROI1.Rotation;
im_rot = imrotate(im,rotation);

fig = figure('Name', 'Set Crop');
hIm2 = imshow(im_rot);
fig.WindowState = 'maximized';

c = drawrectangle('Position',[50,110,sz(2)-100,60],'LineWidth',1,'Color','r'); 
c.Label = 'Set crop area, double click when done';

ROI2 = customWait(c);
close(fig);
crop = round(ROI2.Position);


vid_crop(1,:) = [(crop(2)),(crop(2)+crop(4))].';
vid_crop(2,:) = [(crop(1)),(crop(1)+crop(3))-1].'; %minus1 just to make sure it doesnt exceed boundry


%im_rot = imrotate(im,rotation);
im_croped = im_rot(vid_crop(1,1):vid_crop(1,2),vid_crop(2,1):vid_crop(2,2),:);
% figure('Name', 'Croped and rotated frame')
% imshow(im_croped);


end

function [ROI1, ROI2] = customWait(roi1,roi2)
    % Listen for mouse clicks on the ROI
    l = addlistener(roi1,'ROIClicked',@clickCallback);
    % Block program execution
    uiwait;
    % Remove listener
    delete(l);
    % Return the current position
    ROI1.Position = roi1.Position;
    ROI1.Rotation = roi1.RotationAngle;
    
    if exist('roi2','var')
    ROI2.Position = roi2.Position;
    ROI2.Rotation  = roi2.RotationAngle;
    end

end