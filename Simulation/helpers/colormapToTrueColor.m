function tc = colormapToTrueColor(map,ZData)
% map is a n-by-3 colormap matrix
% ZData is a k-by-w matrix of ZData (or CData, I suppose)
% tc is a kxwx3 Truecolor array based on map and ZData values.
tcIdx = round(rescale(ZData,1,height(map)));
tc = reshape(map(tcIdx,:),[size(ZData),3]);
end
