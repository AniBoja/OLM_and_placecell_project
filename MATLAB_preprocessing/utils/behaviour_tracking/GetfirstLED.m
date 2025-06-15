function [LEDframe, log, lastLEDframe] = GetfirstLED(v, varargin)
% GetfirstLED function finds first frame a blue led flash in video
%
% loads first video frame, asks user to draw roi around LED location
% outputs fig of LEDframe and frame before and after for QC 
% vars... 
% v: a VideoReader object of video to analyse (use VideoReader function)
%
% threshold: LED intensity to detect flash, default = 10
% 
%-----------------------------------------------------------------


P = inputParser;
P.addParameter('threshold',10);
P.addParameter('startframe',1);
P.addParameter('endframe',v.NumFrames);

P.parse(varargin{:});
for i=fields(P.Results)'
   eval([i{1} '=P.Results.(i{1});']); 
end

t = endframe;

% if ~exist('threshold','var')
%     threshold = 10;
% end
% 
% if ~exist('lastframe','var')
%     t = v.NumFrames;
% else 
%     t = lastframe;
% end



vid_frame = read(v,startframe); % read the first frame of the video 

frame1 = figure('Name', 'Select sync LED');
    imshow(vid_frame)
    frame1.WindowState = 'maximized';
    logf('please draw ROI around LED, then double click');
    LEDarea = drawrectangle('Position',[100,100,95,95],'LineWidth',1,'Color','r'); 
    position = customWait(LEDarea);
    logf('LED ROI set');
close(frame1);

p = round(position.Position);

log.cropedy = (p(1)):(p(1)+p(3));
log.cropedx = (p(2)):(p(2)+p(4));


trig = true;
i = startframe;
while trig == true

frame = read(v,i);
    b_frame = frame(log.cropedx,log.cropedy,3);
    log.Avg(i) = mean(b_frame,'all');
    if mean(b_frame,'all') >= threshold
        LEDframe = i;
       
        logf(['First LED flash at frame ' num2str(LEDframe)])
        figure('Name', 'First & Last LEDs')
        subplot(3,2,1)
        imshow((read(v,i-1)))
        subplot(3,2,3)
        imshow((read(v,i)))
        subplot(3,2,5)
        imshow((read(v,i+1)))
        trig = false;
    else
    i = i+1;
    end
end
     
 

trig2 = true;

%t = 100;
while trig2 == true
    frame = read(v,[t-9 t]);
    b_frame = squeeze(frame(log.cropedx,log.cropedy,3,:));
    
    if mean(b_frame,'all') >= threshold
       fin = false;
        for r = t:-1:1 
          frame2 = read(v,r);
          b_frame2 = frame2(log.cropedx,log.cropedy,3);
          if mean(b_frame2,'all') >= threshold
               lastLEDframe = r;
               
                logf(['Last LED flash at frame ' num2str(lastLEDframe)])
                subplot(3,2,2)
                imshow((read(v,r-1)))
                subplot(3,2,4)
                imshow((read(v,r)))
                subplot(3,2,6)
                imshow((read(v,r+1)))
                fin = true;
          end
          if fin==true
          break;
          end
       end
       trig2 = false;
    else
    t = t-10;
    end
end
     

end



function logf(varargin)
    message = sprintf(varargin{1}, varargin{2:end});
    str = ['[' datestr(now(), 'HH:MM:SS') '] ' message];
    disp(str);
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
   
            
           

