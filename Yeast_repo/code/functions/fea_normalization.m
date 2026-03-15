function output = fea_normalization(input, options)
% choice of feature normalization
% the default is 'min-max'
% each row is a sample
% each column is a feature
min_row = options.min_row;
max_row = options.max_row;
nu_norm = options.nu_norm;
method = options.method;
eps = 10^-15;
[n, d] = size(input);

if strcmp(method, 'l2-intra')
    output = input./repmat(sum(input.^2,2).^0.5,1,d);
elseif strcmp(method, 'l2-inter')
    output = input./repmat(sum(input.^2,1).^0.5,n,1);
elseif strcmp(method, 'l2')
    output = input./repmat(sum(input.^2,2).^0.5,1,d);
    output = output./repmat(sum(output.^2,1).^0.5,n,1);
elseif strcmp(method, 'min-max')
    
    output = 2*(input-min_row)./(max_row - min_row + eps)-1;
elseif strcmp(method, 'min-max1')
    
    output = (input-min_row)./(max_row - min_row);

elseif strcmp(method, 'minus-min-l2')
    input = input - repmat(nu_norm,n,1);
    output = input./repmat(sum(input.^2,2).^0.5,1,d);
    %output = input./repmat(sum(abs(input),2),1,d);
elseif strcmp(method, 'l2-minus-min')
    input = input./repmat(sum(input.^2,2).^0.5,1,d);
    nu = mean(input);
    output = input - repmat(nu,n,1);
    
   
    
end
