function [live_data, dead_data] = getFea(method, dead_img_list, live_img_list, cw)
testaug = 'false';
N = 180;    % number of images for each class
% one should pay attention to featrue extraction 
% since glcm, speckles are rotation invariant
% therefore teataug should be 'false'
% densenet features are extracted via pytorch

%% import features
if exist(strcat(cw, '/fea/', method), 'dir') 
%     live_data = importdata(strcat(cw, '/fea/', method, '/live_data.mat'));
%     dead_data = importdata(strcat(cw, '/fea/', method, '/dead_data.mat'));
    live_data = importdata(strcat(cw, '/fea/', method, '/side_live_2017.mat'));
    dead_data = importdata(strcat(cw, '/fea/', method, '/dead_live_2017.mat'));
else
    
    if strcmp(method, 'speckle')
        live_data = extractSpeckle(live_img_list);
        dead_data = extractSpeckle(dead_img_list);
    elseif strcmp(method, 'hog')
        live_data = extractHog(live_img_list, testaug);
        dead_data = extractHog(dead_img_list, testaug);
    elseif strcmp(method, 'glcm')
        live_data = extractGLCM(live_img_list, 'false');
        dead_data = extractGLCM(dead_img_list, 'false');
    end
    
end

% select N = 180 samples to use
% see dead_list_2018.txt
% see live_list_2018.txt
seed = 0;
rand('seed', seed);
no_live = size(live_data, 1);
shuffle_index = randperm(no_live);
live_data = live_data(shuffle_index(1:N), :);


no_dead = size(dead_data, 1);
rand('seed', 6);
shuffle_index = randperm(no_dead);
dead_data = dead_data(shuffle_index(1:N), :);

% live_data = live_data(1:N,:);
% dead_data = dead_data(1:N,:);


