%% figure from the data from the deeplab cut 
function [behave_data] = GetBehaviourData(v)

[file,path] = uigetfile('*.csv');
behave_data = (importdata(fullfile(path,file)));
behave_data = behave_data.data;

certainty_thresh = 0.85;

for i = 1:size(behave_data,1)
    if behave_data(i,4) < certainty_thresh
    behave_data(i,2:4) = nan;
    end
end

location = behave_data(:,2:3);

im_crop = read(v,1);

figure
imshow(im_crop)
hold on 
plot(coord(:,1),coord(:,2))

nan_num = sum(isnan(location(:,1)));

end

