clear all;
clc;
rng('default');

%% set paths
addpath(genpath(pwd));

%% load dataset
LD = load('CMUPIE');
X = LD.X;
Y = LD.Y;
paras.num_cluster = LD.num_class;

paras.label_ratio = 0.1;
paras.alpha = 0.1;
paras.lr = 1e-13;

paras.max_L_left = 5;
paras.max_L_right = 1;

paras.decay_rate = 0.3;
[ACC, MIhat, Purity] = GDNMF(X, Y, paras);




