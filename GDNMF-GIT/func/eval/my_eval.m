function [ACC, MIhat, Purity] = my_eval(H, Y, k)

% H: d*n
% Y: n*1
% k: cluster number

[ACC, MIhat] = evalResults(H, Y);

if iscell(H)
    H = H{numel(H)};
end

label = litekmeans(H',k,'Replicates',100);
[Purity] = compute_purity(label, Y, k);

end

