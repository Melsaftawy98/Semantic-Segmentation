load net
[filename, pathname] = uigetfile('*.*','Pick an image');
I = imread(strcat(pathname,filename));
I = imresize(I,[720 960]);
C = semanticseg(I, net);
B= labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
figure;
imshow(I);
figure;
imshow(B);
pixelLabelColorbar(cmap,classes);