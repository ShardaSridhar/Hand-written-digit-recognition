function [] = naiveBayesGaussian(filename, num_splits, train_percent)
file = csvread(filename);
[N,D] = size(file);% N samples with D-1 features and 1 target variable
k = num_splits;
if strcmp('boston.csv',filename)
       [ boston_50,boston_75 ] = create_datasets(file,N,D); 
       error_rates_B50 = zeros(5,k);
       error_rates_B75 = zeros(5,k);
else
       error_rates = zeros(5,k);
end

indices = crossvalind('Kfold',N,k);

for i = 1:k
        test_indices = find(indices==i);        
        train_indices_total = find(indices~=i);
        for j = 1:length(train_percent)
            train_p_indices = randperm(length(train_indices_total),...
                              floor(train_percent(j)/100*length(train_indices_total)));
            train_indices = train_indices_total(train_p_indices);
            if strcmp(filename,'boston.csv')
                [error_B50] = NBG2class(boston_50,test_indices,train_indices);
                 error_rates_B50(j,i) = error_B50;
                [error_B75] = NBG2class(boston_75,test_indices,train_indices);
                 error_rates_B75(j,i) = error_B75; 
            else
                data=file;
                [error] = NBGkclass(data,test_indices,train_indices);
                error_rates(j,i) = error;
            end
        end
end  
if strcmp('digits.csv',filename)
    pri='Digits- The test set error rates for each training percents are :';
    disp(pri);
    disp(mean(error_rates,2));
    errorbar(mean(error_rates,2),std(error_rates,0,2));
    xlabel('increasing percentage of the training set');
    ylabel('mean and std deviation across different test sets');
    title('error percentage plot for the digits dataset using GNB classifier');
else
    pri='Boston_50- The test set error rates for each training  percents are :';
    disp(pri);
    disp(mean(error_rates_B50,2));
    pri='Boston_75- The test set error rates for each training percents are :';
    disp(pri);
    disp(mean(error_rates_B75,2));
    errorbar(mean(error_rates_B50,2),std(error_rates_B50,0,2));
    xlabel('increasing percentage of the training set');
    ylabel('mean and std deviation across different test sets');
    title('error percentage plot for the boston50 dataset using GNB classifier');
    figure
    errorbar(mean(error_rates_B75,2),std(error_rates_B75,0,2));
    xlabel('increasing percentage of the training set');
    ylabel('mean and std deviation across different test sets');
    title('error percentage plot for the boston75 dataset using GNB classifier');
end   
end

