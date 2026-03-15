function [mask] = getMask(img)
    
I = img;

T = adaptthresh(I, 0.4);
BW1 = imbinarize(I,T);

se = strel('disk',5);
BW1 = imopen(BW1,se);

%   figure
%   imshowpair(I, BW1, 'montage')


[counts,x] = imhist(I,16);
stem(x,counts)

T = otsuthresh(counts);
BW = imbinarize(I,T);
BW = imopen(BW,se);
% figure
% imshow(BW)
%
% imshow(BW & BW1)
m = mean(I(:))*0.8;

BW2 = double(I) > m;
mask = BW & BW1;
mask = BW1 & BW2;

