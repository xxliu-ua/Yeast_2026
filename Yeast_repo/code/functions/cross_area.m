function out = cross_area(imag)
% codes from Wendy to extract speckle features;
% we do not use her codes since the area is not accurate
% we use a segmentation based method instead

out =[];
verbose = 'False';

if numel(size(imag)) == 3
    imag = rgb2gray(imag);
end

Id = imgaussfilt(imag,8);% with standard deviation by sigma = 8
Id = imadjust(Id); %figure; imshow(Id);
I1 = imregionalmax(Id,4);
L1 = bwlabel(I1,8);
num = max(max(L1));


x = []; y = []; %X = []; Y = [];
for j = 1:num
    [x(j,:),y(j,:)] = find(L1 == j,1);
end
if find(x == 1) ~= 0
    k = find(x == 1);
    x(k) = [];
    y(k) = [];
end
if find(y == 1) ~= 0
    k = find(y == 1);
    y(k) = [];
    x(k) = [];
end

[M,N] = size(Id);
local_fwhm = [];
if strcmp(verbose, 'True')
    figure; imshow(Id');
end

corner_points = zeros(length(x), 4, 2);
out_area = zeros(length(x),1);

for ind = 1:length(x)
    L = 0; R = 0; U = 0; D = 0;
    A1 = []; A2 = []; A3 = []; A4 = [];
    for j = 1:y(ind)
        A1(j) = Id(x(ind),j);
    end
    A1 = double(A1);
    if length(A1) < 3
        L = length(A1);
    else
        [pks,locs] = findpeaks(-A1);
        if isempty(pks)
            if min(A1) <= 1/2*double(Id(x(ind),y(ind)))
                c = y(ind);
                while Id(x(ind),c) >= 1/2*double(Id(x(ind),y(ind)))
                    L = L+1;
                    c = c-1;
                    if length(A1) - c ==length(A1)
                        break;
                    end
                end
            else
                L = length(A1);
            end
        else
            
            temp1 = -pks(length(pks));
            % make sure temp1 saves a local max. which is nearest to Id(x(ind),y(ind));
            
            if temp1 > 1/2*double(Id(x(ind),y(ind)))
                c = y(ind);
                for j = 1:length(A1)
                    L = L+1;
                    c = c-1;
                    if Id(x(ind),c) < Id(x(ind),c-1) && Id(x(ind),c) <= Id(x(ind),c+1)
                        break;
                    end
                end
            elseif temp1 == 1/2*double(Id(x(ind),y(ind)))
                c = y(ind);
                while Id(x(ind),c) >= 1/2*double(Id(x(ind),y(ind)))
                    L = L+1;
                    c = c-1;
                    if Id(x(ind),c) == 1/2*double(Id(x(ind),y(ind))) &&Id(x(ind),c-1) ~= 1/2*double(Id(x(ind),y(ind)))
                        break;
                    end
                end
            else
                c = y(ind);
                for j = 1:length(A1)
                    L = L+1;
                    if Id(x(ind),c) >= 1/2*double(Id(x(ind),y(ind))) &&Id(x(ind),c-1) < 1/2*double(Id(x(ind),y(ind)))                        
                        break;
                    end
                    c = c-1;
                end
            end
        end
    end
    for j = 1:(N-y(ind)+1)
        A2(j) = Id(x(ind),y(ind)+j-1);
    end
    A2 = double(A2);
    if length(A2) < 3
        R = length(A2);
    else
        [pks,locs] = findpeaks(-A2);
        if isempty(pks)
            if min(A2) <= 1/2*double(Id(x(ind),y(ind)))
                c = y(ind);
                while Id(x(ind),c) >= 1/2*double(Id(x(ind),y(ind)))
                    R = R+1;
                    c = c+1;
                    if c == length(A2)+y(ind)
                        break;
                    end
                end
            else
                R = length(A2);
            end
        else
            temp2 = -pks(1);
            if temp2 > 1/2*double(Id(x(ind),y(ind)))
                c = y(ind);
                for j = 1:length(A2)
                    R = R+1;
                    c = c+1;
                    if Id(x(ind),c) <= Id(x(ind),c-1) && Id(x(ind),c) <Id(x(ind),c+1)
                        
                        break;
                    end
                end
            elseif temp2 == 1/2*double(Id(x(ind),y(ind)))
                c = y(ind);
                while Id(x(ind),c) >= 1/2*double(Id(x(ind),y(ind)))
                    R = R+1;
                    c = c+1;
                    if Id(x(ind),c) == 1/2*double(Id(x(ind),y(ind))) &&Id(x(ind),c+1) ~= 1/2*double(Id(x(ind),y(ind)))
                        
                        break;
                    end
                end
            else
                c = y(ind);
                for j = 1:length(A2)
                    R = R+1;
                    c = c+1;
                    if Id(x(ind),c) >= 1/2*double(Id(x(ind),y(ind))) && Id(x(ind),c+1) < 1/2*double(Id(x(ind),y(ind)))
                       
                        break;
                    end
                end
            end
        end
    end
    for j = 1:x(ind)
        A3(j) = Id(j,y(ind));
    end
    A3 = double(A3);
    if length(A3) < 3
        U = length(A3);
    else
        [pks,locs] = findpeaks(-A3);
        if isempty(pks)
            if min(A3) <= 1/2*double(Id(x(ind),y(ind)))
                r = x(ind);
                while Id(r,y(ind)) >= 1/2*double(Id(x(ind),y(ind)))
                    U = U+1;
                    r = r-1;
                    if length(A3) - r ==length(A3)
                        break;
                    end
                end
            else
                U = length(A3);
            end
        else
            temp3 = -pks(length(pks));
            if temp3 > 1/2*double(Id(x(ind),y(ind)))
                r = x(ind);
                for j = 1:length(A3)
                    U = U+1;
                    r = r-1;
                    if Id(r,y(ind)) < Id(r-1,y(ind)) && Id(r,y(ind)) <= Id(r+1,y(ind))
                       
                        break;
                    end
                end
            elseif temp3 ==1/2*double(Id(x(ind),y(ind)))
                r = x(ind);
                while Id(r,y(ind)) >= 1/2*double(Id(x(ind),y(ind)))
                    U = U+1;
                    r = r-1;
                    if Id(r,y(ind)) == 1/2*double(Id(x(ind),y(ind))) && Id(r- 1,y(ind)) ~= 1/2*double(Id(x(ind),y(ind)))
                       
                        break;
                    end
                end
            else
                r = x(ind);
                for j = 1:length(A3)
                    U = U+1;
                    if Id(r,y(ind)) >= 1/2*double(Id(x(ind),y(ind))) && Id(r-1,y(ind)) < 1/2*double(Id(x(ind),y(ind)))
                        
                        break;
                    end
                    r = r-1;
                end
            end
        end
    end
    for j = 1:(M-x(ind)+1)
        A4(j) = Id(x(ind)+j-1,y(ind));
    end
    %if isempty(A4)
    %
    D = 0;
    %end
    A4 = double(A4);
    if length(A4) < 3
        D = length(A4);
    else
        [pks,locs] = findpeaks(-A4);
        if isempty(pks)
            if min(A4) <= 1/2*double(Id(x(ind),y(ind)))
                r = x(ind);
                while Id(r,y(ind)) >= 1/2*double(Id(x(ind),y(ind)))
                    D = D+1;
                    r = r+1;
                    if r == length(A4) + x(ind)
                        break;
                    end
                end
            else
                D = length(A4);
            end
        else
            temp4 = -pks(1);
            if temp4 > 1/2*double(Id(x(ind),y(ind)))
                r = x(ind);
                for j = 1:length(A4)
                    D = D+1;
                    r = r+1;
                    if Id(r,y(ind)) <= Id(r-1,y(ind)) && Id(r,y(ind)) <Id(r+1,y(ind))
                        
                        break;
                    end
                end
            elseif temp4 == 1/2*double(Id(x(ind),y(ind)))
                r = x(ind);
                while Id(r,y(ind)) >= 1/2*double(Id(x(ind),y(ind)))
                    D = D+1;
                    r = r+1;
                    if Id(r,y(ind)) == 1/2*double(Id(x(ind),y(ind))) &&Id(r+1,y(ind)) ~= 1/2*double(Id(x(ind),y(ind)))
                        
                        break;
                    end
                end
            else
                r = x(ind);
                for j = 1:length(A4)
                    D = D+1;
                    r = r+1;
                    if Id(r,y(ind)) >= 1/2*double(Id(x(ind),y(ind))) &&Id(r+1,y(ind)) < 1/2*double(Id(x(ind),y(ind)))
                        
                        break;
                    end
                end
            end
        end
    end
    local_fwhm(ind) = 1/2*(R+L)*(D+U);
    if strcmp(verbose, 'True')
        text(x(ind),y(ind),'o','color','g');
    %rectangle('Position',[x(ind)-U y(ind)+R L+R U+D],'EdgeColor','g');
    %text(x(ind),y(ind)-L,'+','color','g');
    %text(x(ind)-U,y(ind),'+','color','g');
    %text(x(ind),y(ind)+R,'+','color','g');
    %text(x(ind)+D,y(ind),'+','color','g');
    hold on;
    plot([x(ind) x(ind)-U],[y(ind)-L y(ind)],'r'); hold on; 
    plot([x(ind) x(ind)+D],[y(ind)-L y(ind)],'r'); hold on; 
        
    plot([x(ind) x(ind)-U],[y(ind)+R y(ind)],'r'); hold on;
    plot([x(ind) x(ind)+D],[y(ind)+R y(ind)],'r');
    end
    X(1,ind) = x(ind);
    Y(1,ind) = y(ind)-L;
    
    corner_points(ind,:,1) = [x(ind),x(ind)-U,x(ind),x(ind)+D];
    corner_points(ind,:,2) = [y(ind)-L, y(ind),y(ind)+R,y(ind)];
    %w(1,ind) = L+R;
    %h(1,ind) = D+U;
    out_area(ind,1) = polyarea(corner_points(ind,:,1),  corner_points(ind,:,2));
    %text(corner_points(ind,:,1), corner_points(ind,:,2),'o','color','g');
end
labels = cellstr(num2str([1:length(x)]'));
dx = 5; dy = 5;
% text(x+dx,y+dy,labels,'Color','green');
% labels = cellstr(num2str([1:length(X)]'));
% dx = 5; dy = 5;
% text(X+dx,Y+dy,labels,'Color','red');

out = out_area;

