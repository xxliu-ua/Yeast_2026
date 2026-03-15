function new_images = rotate(images,k)
% data augmentation operation
% including rotate by 90, 180, 270 flip vertically and horizontally

[H,W,C,N] = size(images);
new_images = uint8(zeros(size(images)));
for i = 1 : N
    if k == 1
        new_images(:,:,:,i) = imrotate(images(:,:,:,i),90);
    elseif k ==2
        new_images(:,:,:,i) = imrotate(images(:,:,:,i),180);
    elseif k ==3
        new_images(:,:,:,i) = imrotate(images(:,:,:,i),270);
    elseif k == 4
        new_images(:,:,1,i) = fliplr(images(:,:,1,i));
        new_images(:,:,2,i) = fliplr(images(:,:,2,i));
        new_images(:,:,3,i) = fliplr(images(:,:,3,i));
    else
        new_images(:,:,1,i) = flipud(images(:,:,1,i));
        new_images(:,:,2,i) = flipud(images(:,:,2,i));
        new_images(:,:,3,i) = flipud(images(:,:,3,i));
    end
end

    
