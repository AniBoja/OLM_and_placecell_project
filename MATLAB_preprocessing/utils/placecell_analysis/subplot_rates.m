function  ax = subplot_rates(toplot, varargin)


P = inputParser;
P.addParameter('font_size',12);
P.addParameter('titles',[]);
P.addParameter('xtitle', 'Location');
P.addParameter('ytitle',  'Place Field');

P.parse(varargin{:});
for i=fields(P.Results)'
   eval([i{1} '=P.Results.(i{1});']); 
end

Plot = toplot;
k = size(toplot,2);

if isempty(titles)
    for t = 1:k 
    titles(t) = strcat("Session ",string(t));
    end
end




fig = figure('Renderer', 'painters', 'Position', [100,100,(400*k),800]);
orient(fig,'landscape');

for i = 1:k
CustomxLabels(1:size(Plot{i},2)) = "" ;
YLabels = 1:size(Plot{i},1);
CustomYLabels = string(YLabels);
CustomYLabels(mod(YLabels,size(Plot{i},1)) ~= 0) = " ";
CustomYLabels(1) = "1";

subplot(1,k,i)
heatmap(Plot{i},'Colormap', parula, 'GridVisible','off','YDisplayLabels',CustomYLabels,'XDisplayLabels',CustomxLabels,'fontsize',font_size);

title(titles(i))
xlabel(xtitle)
ylabel(ytitle)

end