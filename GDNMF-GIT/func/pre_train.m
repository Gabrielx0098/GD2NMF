function [FF, SS_left, SS_right, GG] = pre_train(FF, SS_left, SS_right, GG, paras)

max_L_left = size(SS_left, 2);
max_L_right = size(SS_right, 2);

%% decompose F into next layer
for L = 2:max_L_left
    F_pre = FF{L-1};
    tmp_mat = paras.g_left_inv(F_pre);
    [k1, k2] = size(F_pre);
    k = ceil(paras.decay_rate * min(k1, k2));
%     [FF{L}, SS_left{L}] = nnmf(tmp_mat, k);
    [FF{L}, SS_left{L}] = my_nmf(tmp_mat, k);
end

%% decompose G into next layer
for L = 2:max_L_right
    G_pre = GG{L-1};
    tmp_mat = paras.g_right_inv(G_pre);
    [k1, k2] = size(G_pre);
    k = ceil(paras.decay_rate * min(k1, k2));
%     [SS_right{L}, GG{L}] = nnmf(tmp_mat, k);
    [SS_right{L}, GG{L}] = my_nmf(tmp_mat, k);
end

end

