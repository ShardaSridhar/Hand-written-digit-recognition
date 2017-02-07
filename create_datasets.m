function [ boston_50,boston_75 ] = create_datasets(boston,N,D)
temp=[ones(N,1),boston];
            t_50 = prctile(boston(:,D),50);% 50th percentile
            w_50 = [-t_50,zeros(1,D-1),1]';
            c_50 = sign(w_50'*temp')';
            c_50 = sign(c_50+1);
            boston_50 = [boston(:,1:D-1),c_50];
            
            t_75 = prctile(boston(:,D),75);% 75th percentile
            w_75 = [-t_75,zeros(1,D-1),1]';
            c_75 = sign(w_75'*temp')';
            c_75 = sign(c_75+1);
            boston_75 = [boston(:,1:D-1),c_75];
end

