function [FF, SS_left, SS_right, GG, loss_mat] = fine_tune(FF, SS_left, SS_right, GG, X, paras, Y, Q)

% FF{1} = FF{1} * 0;

num_cluster = paras.num_cluster;

max_L_left = size(SS_left, 2);
max_L_right = size(SS_right, 2);

% initialize cell arrays to store derivatives
dFF = cell(size(FF));
dGG = cell(size(GG));

%% reconstruct FF and GG from out side layers
% for L = max_L_left:-1:2
%     tmp_mat = FF{L} * SS_left{L};
%     FF{L-1} = paras.g_left(tmp_mat);
%     FF{L-1}(FF{L-1}<0) = eps;
% end
% 
% for L = max_L_right:-1:2
%     tmp_mat = SS_right{L} * GG{L};
%     GG{L-1} = paras.g_right(tmp_mat);
%     GG{L-1}(GG{L-1}<0) = eps;
% end

%% initialize auxiliary matrix
% W = rand(num_feature, size(GG{max_L_right}, 1));
% A = X - FF{1} * SS_left{1} * GG{1};
% W = A * GG{max_L_right}' * inv(GG{max_L_right} * GG{max_L_right}');
W = rand(size(X, 1), size(GG{max_L_right}, 1));
% W = zeros(size(X, 1), size(GG{max_L_right}, 1));

P1 = rand(num_cluster, size(GG{1}, 1));
% P1 = zeros(num_cluster, size(GG{1}, 1));
% A = Y * Q;
% B = GG{1} * Q;
% P1 = A * B' * inv(B * B');

P = rand(num_cluster, size(GG{max_L_right}, 1));
% P = zeros(num_cluster, size(GG{max_L_right}, 1));
% A = Y * Q - P1 * GG{1} * Q;
% B = GG{max_L_right} * Q;
% P = A * B' * inv(B * B');

max_iter = 300;
loss_mat = zeros(1, []);
lr = paras.lr;

for iter = 1:max_iter
    
    
    %% update S on layer = 1

%     A = FF{1}' * (FF{1} * SS_left{1} * GG{1} + W * GG{max_L_right}) * GG{1}';
%     B = FF{1}' * X * GG{1}';
%     SS_left{1} = SS_left{1} .* (B ./ (A + eps));
    dS = -FF{1}' * (X - FF{1} * SS_left{1} * GG{1} - W * GG{max_L_right}) * GG{1}';   
    SS_left{1} = PGD(SS_left{1}, dS, lr, 1);
    SS_right{1} = SS_left{1};
    
    dFF{1} = -(X - FF{1} * SS_left{1} * GG{1} - W * GG{max_L_right}) * GG{1}' * SS_left{1}';
    dGG{1} = -(FF{1} * SS_left{1})' * (X - FF{1} * SS_left{1} * GG{1} - W * GG{max_L_right}) - paras.alpha * P1' * (Y - P1 * GG{1} - P * GG{max_L_right}) * Q * Q';
    
    %%
    % W
    A = (FF{1} * SS_left{1} * GG{1} + W * GG{max_L_right}) * GG{max_L_right}';
    B = X * GG{max_L_right}';
    W = W .* (B ./ (A + eps));
    
    % P1
    A =  (P1 * GG{1} + P * GG{max_L_right}) * Q * Q'* GG{1}';
    B = Y * Q * Q' * GG{1}';
    P1 = P1 .* (B ./ (A + eps));
    
    % P
    A = (P1 * GG{1} + P * GG{max_L_right}) * Q * Q' * GG{max_L_right}';
    B = Y * Q * Q' * GG{max_L_right}';
    P = P .* (B ./ (A + eps));
    %%
%     dW = -(X - FF{1} * SS_left{1} * GG{1} - W * GG{max_L_right}) * GG{max_L_right}';
%     W = PGD(W, dW, lr, 0);
%     dP1 = -paras.alpha * (Y - P1 * GG{1} - P * GG{max_L_right}) * Q * Q'* GG{1}';
%     P1 = PGD(P1, dP1, lr, 0);
%     dP = -paras.alpha * (Y - P1 * GG{1} - P * GG{max_L_right}) * Q * Q' * GG{max_L_right}';
%     P = PGD(P, dP, lr, 0);
    
    %% update SS_left
    for i = 2:max_L_left
        dS = FF{i}' * (dFF{i-1} .* paras.g_left_diff(FF{i} * SS_left{i}));
        SS_left{i} = PGD(SS_left{i}, dS, lr, 1);
        
        dFF{i} = (dFF{i-1} .* paras.g_left_diff(FF{i} * SS_left{i})) * SS_left{i}';
    end
    
    %% update SS_right
    for i = 2:max_L_right
        dS = (dGG{i-1} .* paras.g_right_diff(SS_right{i} * GG{i})) * GG{i}';
        SS_right{1} = PGD(SS_right{i}, dS, lr, 1);
        
        dGG{i} = SS_right{i}' * (dGG{i-1} .* paras.g_right_diff(SS_right{i} * GG{i}));
    end
    
    %% update max_layer
    dGG{max_L_right} = dGG{max_L_right} - W' * (X - FF{1} * SS_left{1} * GG{1} - W * GG{max_L_right}) - paras.alpha * P' * (Y - P1 * GG{1} - P * GG{max_L_right}) * Q * Q';
    GG{max_L_right} = PGD(GG{max_L_right}, dGG{max_L_right}, lr, 1);
    
    FF{max_L_left} = PGD(FF{max_L_left}, dFF{max_L_left}, lr, 1);
    
    
    %% reconstruct FF and GG from out side layers
%     for L = max_L_left:-1:2
%         tmp_mat = FF{L} * SS_left{L};
%         FF{L-1} = paras.g_left(tmp_mat);
%         FF{L-1}(FF{L-1}<0) = eps;
%     end
%     
%     for L = max_L_right:-1:2
%         tmp_mat = SS_right{L} * GG{L};
%         GG{L-1} = paras.g_right(tmp_mat);
%         GG{L-1}(GG{L-1}<0) = eps;
%     end
    
    recover_mat = FF{1} * SS_left{1} * GG{1};
    loss = 0.5 * norm(X - recover_mat - W * GG{max_L_right}, 'fro')^2 + 0.5 * paras.alpha * norm((Y - P1 * GG{1} - P * GG{max_L_right}) * Q, 'fro')^2;
    loss_mat = cat(2, loss_mat, loss);
    
    if iter > 1 && abs(loss_mat(iter) - loss_mat(iter-1)) < 1e-4 * loss_mat(iter-1)
        break;
    end
end

end

