function [Pi1, Pi2] = PiArray(k, H, rho, nu, lambda, theta, V0, t, r, d, n, M, N)

    % Here we allow k = log(K/S) be an 2-d matrix.
    u = linspace(1e-12, M, N+1);
    
    k_3d = repmat(k, [1, 1, length(u)]);
    u_3d = reshape(u, 1, 1, []);
    u_3d = repmat(u_3d, [size(k, 1), size(k, 2), 1]);
    cf1 = CFArray(H, rho, nu, lambda, theta, V0, u - 1i, t, r, d, n);
    cf1 = reshape(cf1, length(t), 1, length(u));
    cf1 = repmat(cf1, 1, size(k, 2), 1);

    cf2 = CFArray(H, rho, nu, lambda, theta, V0, u, t, r, d, n);
    cf2 = reshape(cf2, length(t), 1, length(u));
    cf2 = repmat(cf2, 1, size(k, 2), 1);
    
    %cf2 = CFArray(H, rho, nu, lambda, theta, V0, u, t, r, d, n);
    %toc;
    %disp(size(k_3d));
    %disp(size(u_3d));
    %disp(size(cf2));
    integrand2 = real(exp(-1i .* k_3d .* u_3d ) .* (cf2 ./ (1i .* u_3d)));
    %disp(size(integrand2));
    integrand2(:, :, 1) = integrand2(:, :, 1) / 2;
    integrand2(:, :, N+1) = integrand2(:, :, N+1) / 2;
    integrand2 = integrand2 * M / N;
    Pi2 = sum(integrand2, 3) / pi + 1/2;
    
    %cf1 = CFArray(H, rho, nu, lambda, theta, V0, u - 1i, t, r, d, n);
    f_minus1i = CFArray(H, rho, nu, lambda, theta, V0, - 1i, t, r, d, n);
    integrand1 = real(exp(-1i .* k_3d .* u_3d) .* (cf1 ./ (1i .* u_3d .* f_minus1i)));
    integrand1(:, :, 1) = integrand1(:, :, 1) / 2;
    integrand1(:, :, N+1) = integrand1(:, :, N+1) / 2;
    integrand1 = integrand1 * M / N;
    Pi1 = sum(integrand1, 3) / pi + 1/2;

    %disp(size(Pi1));
    %disp(size(Pi2));
end