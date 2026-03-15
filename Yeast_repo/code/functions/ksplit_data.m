function [out_train_set, out_train_labels, out_test_set, out_test_labels] = ksplit_data(dead_data, live_data, indices, k, shuffle)

% the test & train index for the k fold: dead_data
test_index = find(indices == k);
train_index = find(indices ~= k);

% dataset & labels for the current fold
train_set = [live_data(train_index, :); dead_data(train_index, :)];
test_set = [live_data(test_index, :); dead_data(test_index, :)];
train_labels = [ones(length(train_index),1); zeros(length(train_index), 1)];
test_labels = [ones(length(test_index),1); zeros(length(test_index), 1)];


% shuffle the dataset
if strcmp(shuffle, 'true')
    shuffle_index = randperm(size(train_set, 1));
    out_train_set = train_set(shuffle_index, :);
    out_train_labels = train_labels(shuffle_index);
    
    shuffle_index = randperm(size(test_set, 1));
    out_test_set = test_set(shuffle_index, :);
    out_test_labels = test_labels(shuffle_index);
else
    out_train_set = train_set;
    out_train_labels = train_labels;
    out_test_set = test_set;
    out_test_labels = test_labels;
    
end
