function [W, H] = my_nmf(A, k)

[m, n] = size(A);
W = rand(m, k);
H = rand(k, n);

norm_old = 0;
for iter = 1:300
    W = W .* ((A * H') ./ (W * H * H' + eps));
    
    H = H .* ((W' * A) ./ (W' * W * H + eps));
    
    norm_new = norm(A - W * H, 'fro')^2;
    if abs(norm_new - norm_old) < 1e-4 * norm_old
        break;
    end
    norm_old = norm_new;
end

end

