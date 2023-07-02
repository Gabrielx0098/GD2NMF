function [ACC, MIhat, Purity] = GDNMF(X, Y, paras)

% X: data matrix : p * n
% Y: label matrix: c * n;

%% choose non-linear fun
% choose from ['tanh', 'square', 'sigmoid', 'softplus', 'linear']
[paras.g_left, paras.g_left_inv, paras.g_left_diff] = choose_nonlinearfun('square');  
[paras.g_right, paras.g_right_inv, paras.g_right_diff] = choose_nonlinearfun('square');

% the previous name of GDNMF
[~, ~, ~, GG, ~] = run_GDNMF(paras, X, Y);

H = GG{paras.max_L_right};

[ACC, MIhat, Purity] = my_eval(H, Y, paras.num_cluster);

end

