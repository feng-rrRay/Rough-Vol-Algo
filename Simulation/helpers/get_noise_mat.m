function noise_mat_3d_mc = get_noise_mat(J, H, num_path_mc, max_steps_mc)
    %GET_NOISE_MAT Generate noise matrix for hybrid scheme and beyond without
    %incorporating dt information
    Sigma = zeros(J+2,J+2);
    j_vec = 2:J+1; k_vec = 2:J+1;
    Sigma(1,3:J+2) = ((j_vec-1).^(H+1/2) - (j_vec-2).^(H+1/2)) / (H+1/2); 
    F1mat = hypergeom([-H+1/2,1],H+3/2,[1,j_vec]'./[1,k_vec].*([1,j_vec]'<[1,k_vec]));
    Sigma(3:J+2,3:J+2) = ((j_vec'<k_vec).*((j_vec-1)'.^(H+1/2).* (k_vec-1).^(H-1/2) .* F1mat(j_vec-1,k_vec-1) ... 
        -(j_vec-2)'.^(H+1/2).* ([1,1:J-1]).^(H-1/2) .* F1mat([1,1:J-1],[1,1:J-1]))/(H+1/2));
    Sigma = Sigma + Sigma';
    Sigma(logical(eye(J+2))) = [1, 1, ((j_vec-1).^(2*H) - (j_vec-2).^(2*H)) / (2*H)]';
    
    % generate noise matrix (W_i, W_i', W_{i,1},...,W_{i,J}), i = 0,1,...,nT,
    noise_mat_3d_mc = normrnd(0,1,[num_path_mc,max_steps_mc,J+2]);
    L = chol(Sigma)'; % lower triangular matrix; **numerical problem here if H=1/2**
    for j = J+2:-1:1
        noise_mat_3d_mc(:,:,j) = L(j,j) * noise_mat_3d_mc(:,:,j);
        for i = 1:j-1
            noise_mat_3d_mc(:,:,j) = noise_mat_3d_mc(:,:,j) + L(j,i) * noise_mat_3d_mc(:,:,i);
        end
    end

end

