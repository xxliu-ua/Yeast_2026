function [out_acc, out_sen, out_spe, out_auc, out_ratio] = k_percentage(test_labels, test_result, test_set, train_set, gt_ratios)
% use the trained model to predict the ratios of live/dead cells 
% which has been described in the paper
% the experiments are repeated 10 times, and the average value is used as
% the final performance, so as to avoid the effect of data split
% the gt_ratios is an array, representing the ground_truth of the ratios



no_loop = length(gt_ratios);


no_aug = 1;
N = length(test_labels);
test_index = test_labels(1:no_aug:N);
acc = zeros(10,no_loop);
auc = zeros(10,no_loop);
sen = zeros(10,no_loop);
spe = zeros(10,no_loop);
ratio = zeros(10,no_loop);

model = test_result.model;
pos = find(model.Label==1);
clear options
options.method = 'min-max';%'min-max'; % 'min-max', 'l2'
options.min_row = min(train_set);
options.max_row = max(train_set);
options.nu_norm = mean(train_set);
test_set = fea_normalization(test_set, options);

for loop_id = 1 : no_loop
    for i = 1 : 10
        index = find(test_index==0);
        select_index = randperm(length(index));
        xttt = round(length(index)*gt_ratios(loop_id));
        select_id = index(select_index(1:xttt));
        select_dead_index = zeros(length(select_id)*no_aug,1);
        for j = 1 : length(select_id)
            select_dead_index((j-1)*no_aug+1:j*no_aug) = (select_id(j)-1)*no_aug+1:select_id(j)*no_aug;
        end
        live_index = find(test_labels == 1);
        whole_index = [live_index;select_dead_index];
        
        new_test_data = test_set(whole_index, :);
        new_test_label = test_labels(whole_index);
        
        [~, ~, prob_estimates] = svmpredict2(new_test_label, double(new_test_data), model, '-b 1');
        prob_estimates = reshape(prob_estimates(:,pos),[no_aug, length(new_test_label)/no_aug]);
        prob_estimates = mean(prob_estimates,1)';
        predict_label = double(prob_estimates>0.5);
        
        
        new_test_id = new_test_label(1:no_aug:length(new_test_label));
        auc(i, loop_id) = roc_curve(prob_estimates,new_test_id);
        auc(i, loop_id) =  auc(i, loop_id)*100;
        acc(i, loop_id) = sum(predict_label == new_test_id)/length(predict_label)*100;
        tp = sum( double(predict_label & new_test_id));
        sen(i, loop_id) = tp/sum(new_test_id == 1)*100;
        tn = sum(double(~predict_label & ~new_test_id));
        spe(i, loop_id) = tn/sum(new_test_id == 0)*100;
        ratio(i, loop_id) = sum(predict_label==0)/sum(predict_label==1);
    end
end
out_acc = mean(acc);
out_spe = mean(spe);
out_sen = mean(sen);
out_auc = mean(auc);
out_ratio = mean(ratio);
