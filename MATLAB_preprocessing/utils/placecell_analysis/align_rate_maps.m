function [aligned_rates, aligned_by_rates, aligned_by_IDs] = align_rate_maps(to_align_rates_id, align_by_rates_id)

to_align_rates = to_align_rates_id{1};
to_align_IDs = to_align_rates_id{2};

align_by_rates = align_by_rates_id{1};
align_by_IDs = align_by_rates_id{2};


numberbins = size(to_align_rates,2);

%[common_cells, index_common, index_common2] = intersect(to_align_IDs,align_by_IDs,'sorted');
%p = find(~ismember(align_by_IDs,common_cells)); %find the index of cells in all cells not in place cells

[~, index_common] = ismember(align_by_IDs, to_align_IDs);
p = find(~index_common);

index_common(index_common==0)= [];

Rates_to_arrange = to_align_rates(index_common,:);
IDs_to_arrange = to_align_IDs(index_common,:);

New_Rates_array = zeros(size(align_by_IDs,1),numberbins);
ID_array = string;

index = (1:size(align_by_IDs,1));
new_index = index;
new_index(p) =[];



for i = 1:size(new_index,2) 
    
    New_Rates_array(new_index(i),:) = Rates_to_arrange(i,:);
    %ID_array(new_index(i)) = IDs_to_arrange(i);
     
end



to_be_sorted_rates = New_Rates_array;


rates = align_by_rates;

maxrate_index = zeros(size(rates,1),1);
for i = 1:size(rates,1) 
[~, idx] = max(rates(i,:));
maxrate_index(i) = idx;
end 

[~,id]=sort(maxrate_index);

aligned_by_rates =rates(id,:);
aligned_rates = to_be_sorted_rates(id,:);

aligned_by_IDs = align_by_IDs(id);


end
