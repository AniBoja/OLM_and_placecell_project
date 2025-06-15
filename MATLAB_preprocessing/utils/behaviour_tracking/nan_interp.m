function X_interp = nan_interp(X)

for p = 1:size(X,2)    
t = X(:,p);    
idx = ~isnan(t);
X_interp(:,p) = interp1(find(idx),t(idx),(1:numel(t))');
end

end