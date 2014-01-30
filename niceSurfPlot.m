%%

% the data that you want to plot as a 3D surface.
[x,y,z] = peaks;
 
% get the corners of the domain in which the data occurs.
min_x = min(min(x));
min_y = min(min(y));
max_x = max(max(x));
max_y = max(max(y));
 
% the image data you want to show as a plane.
planeimg = abs(z);
 
% scale image between [0, 255] in order to use a custom color map for it.
minplaneimg = min(min(planeimg)); % find the minimum
scaledimg = (floor(((planeimg - minplaneimg) ./ ...
    (max(max(planeimg)) - minplaneimg)) * 255)); % perform scaling
 
% convert the image to a true color image with the jet colormap.
colorimg = ind2rgb(scaledimg,jet(256));
 
% set hold on so we can show multiple plots / surfs in the figure.
figure; hold on;
 
% do a normal surface plot.
surf(x,y,z,'edgecolor','none');
 
% set a colormap for the surface
colormap(gray);
 
% desired z position of the image plane.
imgzposition = -10;
 
% plot the image plane using surf.
surf([min_x max_x],[min_y max_y],repmat(imgzposition, [2 2]),...
    colorimg,'facecolor','texture')
 
% set the view.
view(45,30);
 
% label the axes
xlabel('x');
ylabel('y');
zlabel('z');