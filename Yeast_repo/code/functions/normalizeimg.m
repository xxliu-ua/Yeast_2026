function new_img = normalizeimg(img)
% normalize the images to [0, 255]
    img_min = min(img(:));
    img_max = max(img(:));
    new_img = 255*(img - img_min)./(img_max - img_min);
    new_img = uint8(new_img);