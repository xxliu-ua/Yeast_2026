function [eacc,eauc, acc, auc, para] = opt_linear_svm(train_set,train_labels)
% parameter selection of C and g using 5-fold cross-fold validation
% average auc is used as the metric
% for parameter selection, the shuffle should set to be false in the
% function "split_data"
testaug = 'false';

if strcmp(testaug,'true') 
    no_aug = 6;    
elseif strcmp(testaug, 'flip')
    no_aug = 2;  
else
    no_aug = 1;
end 

[N, dim] = size(train_set);
N = N/no_aug;

%% Parameters can be tuned
gmax = -6;  % -1
gmin = -10;  % libsvm -15
cmax = 4;   % libsvm 15 % default 4
cmin = 0;
W = [1];

%%
C = [cmin:cmax];
G = [gmin:gmax];

basenum = 2;
v = 5;

% record acc & auc with different c & g,and find the bestauc with the smallest c
bestc = 1;
bestg= 0.07;
bestauc = 0;

m = length(C);
n = length(G);
nn = length(W);

acc = zeros(m,n,nn,v);
auc = zeros(m,n,nn,v);
eacc = zeros(m,n,nn,v);
eauc = zeros(m,n,nn,v);
indices  = crossvalind('Kfold',N,v);
para = {};
%% select the best parameters
for i = 1:m
    para.c = basenum^C(i);
    for j = 1:n
        para.g = basenum^G(j);
        for p = 1:length(W)
            para.w = W(p);
            for kk = 1 : v
                val = find(indices == kk);         
                val_index = zeros(length(val)*no_aug, 1);
                for iii = 1 : length(val) 
                    val_index((iii-1)*no_aug+1:iii*no_aug) = (val(iii)-1)*no_aug+1:val(iii)*no_aug;
                end
                val_set = train_set(val_index, :);    
                val_labels = train_labels(val_index);
                
                train_subset_index = setdiff(1:N*no_aug, val_index);
                train_subset = train_set(train_subset_index, :);
                train_subset_labels= train_labels(train_subset_index);
                [test_result, ensemble_test_result] = linear_svm_classifier(train_subset,val_set,train_subset_labels,val_labels,para,testaug);
                acc(i,j,p,kk) = test_result.acc; 
                auc(i,j,p,kk) = test_result.auc;
%                 eacc(i,j,p,k) = ensemble_test_result.acc;
%                 eauc(i,j,p,k) = ensemble_test_result.auc;
%                 
                
                
            end
        end
             
    end
end

mean_acc = mean(acc, 4);
[r,col] = find(mean_acc == max(mean_acc(:)));
para.c = 2^(C(r(1)));
para.g = 2^(G(col(1)));
para.w = 1;

