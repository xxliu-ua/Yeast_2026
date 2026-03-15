% classification for scattering images of yeast cells

clear;


%% parameters------------------------------------------------------

N = 150;                % number of samples of each group % set this to lowest batch size
K = 5;                  % 5 fold cross-validation
testaug = 'false';      % we did not use test augmentation 
shuffle = 'true';       % shuffle the data, it does not affect the performance much
fea = 'densenet';       % densenet features are used
% cw = pwd;               % the current work directory


%% import data

tStart = tic;

y1_data = importdata(strcat('yeast_species1.csv')); % feature location for species 1
y2_data = importdata(strcat('yeast_species2.csv')); % feature location for species 2

y1_data = [y1_data];
y2_data = [y2_data];

% randomly generate the indices for the k-fold cross-validation

indices = 1:K;
indices = repmat(indices, 1, N/K);
rand('seed', 5); 
indices = indices(randperm(N));

%% train & test
acc = zeros(K, 1);
spe = zeros(K, 1);
sen = zeros(K, 1);
auc = zeros(K, 1);
ratio = zeros(K, 1);

for k = 1 : K
    % get the train & test data for the k's fold
    [train_set, train_labels, test_set, test_labels] = ksplit_data(y1_data, y2_data, indices, k, shuffle);
    
     % best parameter selection
     % select the best c & g of SVM
     %%% change optimizer if using RBF or linear SVM !!!
    [eacc1,eauc1, acc1, auc1, para] = opt_linear_svm(train_set,train_labels);
    
    
    %%% change classifier if using RBF or linear SVM !!!
    [test_result, ensemble_test_result, prob] = linear_svm_classifier(train_set,test_set,train_labels,test_labels,para,testaug);
    acc(k) = test_result.acc;
    sen(k) = test_result.sen;
    spe(k) = test_result.sp;
    auc(k) = test_result.auc;
    
end
disp('acc');
disp(num2str(mean(acc)));
disp('sen');
disp(num2str(mean(sen)));
disp('spe');
disp(num2str(mean(spe)));
disp('auc');
disp(num2str(mean(auc)));
disp('ratio');



toc(tStart)
%beep
