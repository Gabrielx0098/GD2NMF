function [FF, SS_left, SS_right, GG, loss_mat] = run_GDNMF(paras, X, Y)

% the g(G + f(G))
% X = paras.g_left_inv(X);

num_sample = size(X, 2);
num_cluster = paras.num_cluster;

%% construct Y_hat
Y_mat = zeros(num_cluster, num_sample);
for j = 1:num_sample
    i = Y(j);
    Y_mat(i, j) = 1;
end

q = ceil(paras.label_ratio * num_sample);
Q = blkdiag(eye(q), zeros(num_sample-q));

%% initialization
[p, n] = size(X);
k1 = ceil(paras.decay_rate * min(p, n));
k2 = k1;
[FF, SS_left, SS_right, GG] = initialize(X, k1, k2, paras);

%% pre-train
[FF, SS_left, SS_right, GG] = pre_train(FF, SS_left, SS_right, GG, paras);

%% fine-tune
loss_mat = [];
[FF, SS_left, SS_right, GG, loss_mat] = fine_tune(FF, SS_left, SS_right, GG, X, paras, Y_mat, Q);

end
