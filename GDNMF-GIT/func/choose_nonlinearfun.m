   function [g, g_inv, g_diff] = choose_nonlinearfun(nonlinearity)

if strcmp(nonlinearity, 'tanh') == 1
    g = @(x) 1.7159*tanh(1.5*x);
    g_inv = @(x) atanh(x);
    g_diff = @(x) 1 - tanh(x).^2;
elseif strcmp(nonlinearity, 'square') == 1  
    g = @(x) x.^2;
    g_inv = @(x) x.^0.5;
    g_diff = @(x) 2 .* x;
elseif strcmp(nonlinearity, 'sigmoid') == 1
    sigmoid = @(x) (1./(1+exp(-x)));
    g = sigmoid;
    g_inv = @(x) log(x ./ (1 - x));
    g_diff = @(x) sigmoid(x) .* (1 - sigmoid(x));
elseif strcmp(nonlinearity, 'softplus') == 1
    g = @(x) log(1 + exp(x));
    g_inv = @(x) log(exp(x) - 1);
    g_diff = @(x) exp(x) ./ (1 + exp(x));
else
    g = @(x) x;
    g_inv = @(x) x;
    g_diff = @(x) 1;
end

end

