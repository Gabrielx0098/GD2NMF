% this code is used for measure purity of clustering 
% written on June 31st by Peng Zhao 
function [purity] = compute_purity(recover, truth, numCluster)

PosiCluster = cell(1,numCluster);
ActualCluster = cell(1,numCluster);
num = zeros(1,numCluster);
for i = 1: numCluster
    PosiCluster{i} = find(recover == i);
    ActualCluster{i} = truth(PosiCluster{i});
    num(1,i) = max(histc(ActualCluster{i},1:max(ActualCluster{i})));
end
purity = 1/length(truth)*sum(num);

end


