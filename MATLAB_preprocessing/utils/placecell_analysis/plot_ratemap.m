function [sorted_rates, ax] = plot_ratemap(rates,varargin)

%% function to sort and plot out hte rate maps for each cell. 
%will sort based on the peak rate and the centroid to output two figures


P = inputParser;
P.addParameter('FontSize',12);
P.addParameter('xtitle', 'Location');
P.addParameter('ytitle',  'Place Field');
P.addParameter('cmap',  parula);
P.addParameter('plottitle',  ' ');
P.addParameter('addxline',  []);
P.addParameter('xlinecolor',  "red");
P.addParameter('figoutput',true,@islogical);

P.parse(varargin{:});
for i=fields(P.Results)'
   eval([i{1} '=P.Results.(i{1});']); 
end

 
bin_size = size(rates,2);
  
plottitle = string(plottitle);

centroid = double.empty;
maxrate_index = double.empty;

for i = 1:size(rates,1) 
cent = regionprops(true(size(rates(i,:))),rates(i,:),  'WeightedCentroid');
centroid(i) = cent.WeightedCentroid(1);

[~, idx] = max(rates(i,:));
maxrate_index(i) = idx;
end 

[~,id2]=sort(centroid);
[~,id]=sort(maxrate_index);

rates_sorted=rates(id,:);
rates_sorted_cent=rates(id2,:);

%filtering if want to 

% bins = linspace(1,number_bins,number_bins);
% for i = 1:size(rates_sorted,1)
% f1(i,:)= gaussfilt(bins,rates_sorted(i,:),1);
% f2(i,:)= gaussfilt(bins,rates_sorted_cent(i,:),1);
% end

f1 = rates_sorted;
f2 = rates_sorted_cent;
sorted_rates = f1;



CustomxLabels(1:bin_size) = "" ;
YLabels = 1:size(f1,1);
CustomYLabels = string(YLabels);
CustomYLabels(mod(YLabels,size(f1,1)) ~= 0) = " ";
CustomYLabels(1) = "1";

if figoutput
    figure('Renderer', 'painters', 'Position', [400,400,200,489])
end
    ax = heatmap(f1,'Colormap', cmap, 'GridVisible','off','YDisplayLabels',CustomYLabels,'XDisplayLabels',CustomxLabels,'fontsize',FontSize);
    if ~isempty(addxline)
        for i=1:size(addxline,2)
            plotLineThroughColumnNr = addxline(i);
            nColumn = numel(ax.XData);
            columnWidth = (ax.InnerPosition(3))/(nColumn);
            lineXCoord = (nColumn-(nColumn-plotLineThroughColumnNr+0.5))*columnWidth + ax.InnerPosition(1);
            annotation("line",[lineXCoord lineXCoord],[ax.InnerPosition(2) ax.InnerPosition(2)+ax.InnerPosition(4)],"Color",xlinecolor,'LineWidth',0.25, 'LineStyle','--')
        end
    end

    title(plottitle)

if figoutput
    figure('Renderer', 'painters', 'Position', [700,400,200,489])
    heatmap(f2,'Colormap', cmap, 'GridVisible','off','YDisplayLabels',CustomYLabels,'XDisplayLabels',CustomxLabels,'fontsize',FontSize)
    title(plottitle)
end
end