function [ error ] = NBGkclass( data,test_indices,train_indices)
[N,d] = size(data);
labels = sort(unique(data(:,d)))';
test = data(test_indices,:);
n_test = length(test_indices);

train = data(train_indices,:);
n_train = length(train_indices);
train_labels = train(:,d);
test_labels = test(:,d);

% Compute class priors
n = zeros(1,length(labels));
mu = zeros(length(labels),d-1);
covar = zeros(d-1);
for i = 1:length(labels)
    n(i) = sum(train(:,d)==labels(i));
    C = train(train_labels==labels(i),1:d-1);
    mu(i,:) = mean(C);
    covar = covar +n(i)/n_train*cov(C);
end
p = n/n_train;
w = zeros(d-1,length(labels));
w0 = zeros(1,length(labels));
for i =1:length(labels)
    w(:,i) = pinv(covar)*mu(i,:)';
    w0(i) = -1/2*mu(i,:)*pinv(covar)*mu(i,:)'+log(p(i));
end
a = w'*test(:,1:d-1)'+w0'*ones(1,n_test);
pred_label = zeros(1,n_test);
for i=1:n_test

    [~,pred_label(i)] = max(exp(a(:,i))/sum(exp(a(:,i))));

end
pred_label=pred_label-1;

error=mean(pred_label'~=test_labels)*100;

end

