function [Binframe] = threshold_select(v,color, frame, threshold,varargin)


P = inputParser;
P.addParameter('foregroundDetect',false,@islogical)
P.addParameter('dilate',false,@islogical)

P.parse(varargin{:});
for i=fields(P.Results)'
   eval([i{1} '=P.Results.(i{1});']); 
end


%can use with a live script to select the treshold that gives the best
%setting
if foregroundDetect
    foregroundDetector = vision.ForegroundDetector('NumTrainingFrames',10,...
                    'InitialVariance',0.01);
    for i = 1:20 % do initial training on the first frames of the video
        rgbFrame = read(v,i);
        foregroundMask = step(foregroundDetector,im2gray(im2single(rgbFrame)));
    end
end

if color == 'r'
    color_num = 1;
end
if color == 'b'
    color_num = 3;
end
if color == 'g' 
    color_num = 2;
end
    rgbFrame = read(v,frame); % Acquire single frame
   
   
    %split the Channel    
    diffFrame = imsubtract(rgbFrame(:,:,color_num), rgb2gray(rgbFrame)); % Get red component of the image
    if foregroundDetect
        foregroundMask = step(foregroundDetector,im2gray(im2single(rgbFrame)));
        diffFrame(~foregroundMask)=0;
    end
    if dilate
       se = offsetstrel('ball',3,3);
       diffFrame = imdilate(diffFrame,se);
    end
    %diffFrame = medfilt2(diffFrame, [3 3]); % Filter out the noise by using median filter
    binFrame = imbinarize(diffFrame, threshold); % Convert the image into binary image with the red objects as white
    
   Binframe =  binFrame;
end