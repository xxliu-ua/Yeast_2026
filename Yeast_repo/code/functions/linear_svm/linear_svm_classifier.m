function [test_result, ensemble_test_result,prob_estimates] = linear_svm_classifier(train_set,test_set,train_index,test_index,para,testaug)

% data normalization
clear options
options.method = 'min-max';%'min-max'; % 'min-max', 'l2'
options.min_row = min(train_set);
options.max_row = max(train_set);
options.nu_norm = mean(train_set);
train_set = fea_normalization(train_set, options);
train_set = sparse(train_set);
test_set= fea_normalization(test_set, options);
test_set = sparse(test_set);
No_test = size(test_set,1);
No_train = size(train_set,1);
test_label = test_index;

% train svm
c = para.c;
g = para.g;
w = para.w;
cmd = ['-c ',num2str(c),...
    ' -w1 ', num2str(w), '-b 1'];
model = train(train_index, double(train_set), cmd);
pos = find(model.Label==1);
if pos == 2
    posit = 1;
end

% test
test_result = {};
ensemble_test_result = {};

if strcmp(testaug, 'false')
    tic;
    [predict_label, accuracy, prob_estimates] = predict(test_index, double(test_set), model); % test the training data
    if pos == 2
        prob_estimates = prob_estimates(:,posit);
        predict_label = double(prob_estimates<0.5);
    else
        prob_estimates = prob_estimates(:,pos);
    %prob_estimates = round(prob_estimates, 3);
        predict_label = double(prob_estimates>0.5);
    end
    
    toc
else
    
    [predict_label, accuracy, prob_estimates] = predict(test_index, double(test_set), model); % test the training data
    prob_estimates = reshape(prob_estimates(:,pos),[n, No_test/n]);
    prob_estimates = mean(prob_estimates,1)';
    %prob_estimates = round(prob_estimates, 3);
    predict_label = double(prob_estimates<0.5);
    
    ensemble_test_result.predict_label = predict_label;
    ensemble_test_result.acc = accuracy(1);
    ensemble_test_result.prob_estimates = prob_estimates;
    ensemble_test_result.auc = roc_curve(prob_estimates,test_label);
    ensemble_test_result.auc = ensemble_test_result.auc *100;
    ensemble_test_result.model = model;
    tp = sum(double(predict_label & test_label));
    ensemble_test_result.sen = tp/sum(test_label == 1)*100;
    tn = sum(double(~predict_label & ~test_label));
    ensemble_test_result.sp = tn/sum(test_label== 0)*100;
    
    
    clear predict_label; clear accuracy; clear prob_estimates;
    [predict_label, accuracy, prob_estimates] = predict(test_index(1:no_aug: No_test), double(test_set(1:n: No_test,:)), model);
    prob_estimates = prob_estimates(:,pos);
    %prob_estimates = round(prob_estimates, 3);
    
end


test_result.predict_label = predict_label;
test_result.prob_estimates = prob_estimates;
test_result.model = model;
if pos ==2
    test_result.auc = lin_roc_curve_posit(prob_estimates,test_label);
else
    test_result.auc = lin_roc_curve_pos(prob_estimates,test_label);
end

test_result.auc =  test_result.auc*100;
test_result.acc = accuracy(1);
tp = sum( double(predict_label & test_label));
test_result.sen = tp/sum(test_label == 1)*100;
tn = sum(double(~predict_label & ~test_label));
test_result.sp = tn/sum(test_label == 0)*100;