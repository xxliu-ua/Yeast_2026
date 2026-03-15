function [new_img] = enhance_img(img, method)
% new_img: the log scale image
if strcmp(method, 'log')
    img = double(img);
    max_i = max(img(:));
    ratio = 255/log10(1+max_i);
    new_img = ratio.*log10(1+img);
    new_img = uint8(new_img);
end

