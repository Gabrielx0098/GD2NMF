function [FF, SS_left, SS_right, GG] = initialize(data_mat, k1, k2, paras)

max_L_left = paras.max_L_left;
max_L_right = paras.max_L_right;

FF = cell(1, max_L_left);
GG = cell(1, max_L_right);
SS_left = cell(1, max_L_left);
SS_right = cell(1, max_L_right);

% [F_1, S_1, G_1, ~] = nmtf(data_mat, k1, k2);

[F_1, tmp_mat] = my_nmf(data_mat, k1);
[S_1, G_1] = my_nmf(tmp_mat, k2);

% [F_1, tmp_mat] = nnmf(data_mat, k1);
% [S_1, G_1] = nnmf(tmp_mat, k2);

FF{1} = F_1;
GG{1} = G_1;
SS_left{1} = S_1;
SS_right{1} = S_1;

end

