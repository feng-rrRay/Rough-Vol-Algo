function noise_mat_new = resize_noise_mat(noise_mat_3d_mc,num_steps_mc)
    %RESIZE_NOISE_MAT Resize the noise matrix with coupling strategy employed
    num_path_mc     = size(noise_mat_3d_mc, 1);
    d               = size(noise_mat_3d_mc, 3);
    num_noise_steps = size(noise_mat_3d_mc, 2);
    if num_noise_steps > num_steps_mc
        if mod(num_noise_steps, num_steps_mc) == 0
            noise_mat_new = zeros(num_path_mc, num_steps_mc, d);
            chunk_size = floor(num_noise_steps / num_steps_mc);
            for i = 1 : num_steps_mc
                noise_mat_new(:, i, :) = sum(noise_mat_3d_mc(:, (i-1)*chunk_size+1 : i*chunk_size, :), 2) ./ sqrt(chunk_size);
            end
        else
            disp("Noise matrix size is not compatible with the number of time steps!");
        end
    else
        noise_mat_new = noise_mat_3d_mc;
    end
end

