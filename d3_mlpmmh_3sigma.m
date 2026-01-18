close all;
clear;
clc;
format long


T = 50;
particle_count = 50;
law_particles_count = 50;
Y_obs = readmatrix(['d3_obs_T_50N.txt']);
params = [0, 0.5, 0.3, 0.4, 0.4, 0.05];
d = 3;
Lmin = 4;
LP = 5;

hl = zeros(LP - Lmin + 1, 1);
hl(1) = 2^(-Lmin);
for l = 1:LP- Lmin
    hl(l+1) = 2^(-l- Lmin);
end

%number of iterations for each level
Nl = [10000,10000];
%store the acceptance rate
Aln = zeros(1, 1);
Theta_trace = cell(LP - Lmin + 1, 1);
Theta_trace_1 = cell(LP - Lmin,1);
Theta_trace_2 = cell(LP - Lmin,1);

Theta_traceN = cell(LP - Lmin + 1, 1);
Theta_trace_1N = cell(LP - Lmin,1);
Theta_trace_2N = cell(LP - Lmin,1);

%mean of theta over iterations
ML_Theta_trace = cell(LP - Lmin + 1, 1);
%weights for finer and corse level        
H1_trace = cell(LP - Lmin, 1);
H2_trace = cell(LP - Lmin, 1);

for k = 1 : LP - Lmin + 1
    Theta_trace{k, 1} = zeros(Nl(k),9);
    Theta_traceN{k,1} = zeros(Nl(k),9);
    ML_Theta_trace{k, 1} = zeros(Nl(k),9);
end

for i = 1:LP - Lmin
    Theta_trace_1{i,1} = zeros(Nl(i+1),9);
    Theta_trace_2{i,1} = zeros(Nl(i+1),9);

    Theta_trace_1N{i,1} = zeros(Nl(i+1),9);
    Theta_trace_2N{i,1} = zeros(Nl(i+1),9);

    H1_trace{i,1} = zeros(Nl(i+1),1);
    H2_trace{i,1} = zeros(Nl(i+1),1);
end


delta = 2^(-Lmin);
Theta_A =  [0.7, 0.4, -2,-1.9,-1.2, -2, -0.1, -1, -2.2];
tic;
Theta_A_p = Theta_A;
Theta_SIG_p = [Theta_A_p(1), Theta_A_p(2), exp(Theta_A_p(3)), exp(Theta_A_p(4)),exp(Theta_A_p(5)),exp(Theta_A_p(6)),exp(Theta_A_p(7)),exp(Theta_A_p(8)),exp(Theta_A_p(9))];

X0 = zeros(d,particle_count);
X0 = sample_X0(params, particle_count);

X_measure = simulate_law_model(delta, T, X0, Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),Theta_SIG_p(5),Theta_SIG_p(6));
Z =  particle_filter(Y_obs, X_measure, delta, T, X0,Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),Theta_SIG_p(5),Theta_SIG_p(6),Theta_SIG_p(7),Theta_SIG_p(8),Theta_SIG_p(9));

lZ = Z;
l_pos_Theta_A_p = l_posterior(Theta_A_p, lZ);
Theta_trace{1,1}(1,:) = Theta_A_p;

N_count_1 = 0;
N_count_last_1 = 0;

Sigma_A1 = 0.225*diag([1.25,1.1,1.1,1.15,1.2,1.15,1,1.1,1.2]);
Sigma_A = 0.2*diag([1.3,1.1,1.1,1.15,1.2,1.15,1,1.1,1.2]);


for iter = 1:Nl(1)
 
    if mod(iter, 100) == 0
        fprintf('iter = %f\n', iter);
        disp(['current AC = ', num2str(N_count_1/(iter))]);
        disp(['current new AC = ', num2str((N_count_1 - N_count_last_1)/(mod(iter,50)+1))]);
        disp(['current estimate = [', num2str(Theta_A_p(1)),  ', ' , num2str(Theta_A_p(2)), ', ',num2str(Theta_A_p(3)),  ', ', num2str(Theta_A_p(4)),  ', ' num2str(Theta_A_p(5)) ,  ', ', num2str(Theta_A_p(6)),  ', ' num2str(Theta_A_p(7)) ,  ', ', num2str(Theta_A_p(8)),  ', ' num2str(Theta_A_p(9)) ']'])
        disp(['estimated posterior = ', num2str(l_pos_Theta_A_p)]);
    end
    if mod(iter, 100) == 0
        N_count_last_1 = N_count_1;
    end
    
    Theta_A_prime_1 = Theta_A_p;
    Theta_A_prime = mvnrnd(Theta_A_prime_1, Sigma_A1*Sigma_A1');
    Theta_SIG_prime = [ Theta_A_prime(1), Theta_A_prime(2), exp(Theta_A_prime(3)), exp(Theta_A_prime(4)),exp(Theta_A_prime(5)), exp(Theta_A_prime(6)),exp(Theta_A_prime(7)), exp(Theta_A_prime(8)),exp(Theta_A_prime(9))];

    X_measure_prime = simulate_law_model(delta, T, X0, Theta_SIG_prime(1),Theta_SIG_prime(2),Theta_SIG_prime(3),Theta_SIG_prime(4),Theta_SIG_prime(5),Theta_SIG_prime(6));
    Z_prime =  particle_filter(Y_obs, X_measure_prime, delta, T, X0,Theta_SIG_prime(1),Theta_SIG_prime(2),Theta_SIG_prime(3),Theta_SIG_prime(4),Theta_SIG_prime(5),Theta_SIG_prime(6),Theta_SIG_prime(7),Theta_SIG_prime(8),Theta_SIG_prime(9));
    lZ_prime = Z_prime;
    l_pos_Theta_A_prime = l_posterior(Theta_A_prime, lZ_prime);
 
    alpha_U = min(0, l_pos_Theta_A_prime - l_pos_Theta_A_p);
    U = log(rand);
    
    if U < alpha_U
        Theta_A_p = Theta_A_prime;
        Theta_SIG_p = Theta_SIG_prime;
        X_measure = X_measure_prime;
        lZ = lZ_prime;
        l_pos_Theta_A_p = l_pos_Theta_A_prime;
        Theta_trace{1, 1}(iter,:) = Theta_A_prime; 
        N_count_1 = N_count_1 + 1;        
    else
        Theta_trace{1, 1}(iter,:) = Theta_A_p; 
        lZ = particle_filter(Y_obs, X_measure, delta, T, X0,Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),Theta_SIG_p(5),Theta_SIG_p(6),Theta_SIG_p(7),Theta_SIG_p(8),Theta_SIG_p(9));
        l_pos_Theta_A_p2 = l_posterior(Theta_A_p, lZ);  
        if true && (l_pos_Theta_A_p2 ~= -inf)
            l_pos_Theta_A_p = l_pos_Theta_A_p2;
        end        
    end 

end

Aln = N_count_1 / Nl(1);
H1_sum = 0;
H2_sum = 0;
tic;

%mlpmmh
for l = 1:LP - Lmin 

    level = l + Lmin;
    fprintf('level = %f\n', level);
    delta_t = 1/2^(level);
    
    X0_m = zeros(d,law_particles_count);
    X0_m = sample_X0(params, law_particles_count);

    X0_pf = zeros(d,particle_count);
    X0_pf = sample_X0(params, particle_count);

   Theta_l = mean(Theta_trace{1,1});
   %Theta_l = [0.7, 0.4, -2,-1.9,-1.2, -2, -0.1, -1, -2.2];
   Theta_SIG_l = [Theta_l(1), Theta_l(2), exp(Theta_l(3)), exp(Theta_l(4)),exp(Theta_l(5)),exp(Theta_l(6)),exp(Theta_l(7)),exp(Theta_l(8)),exp(Theta_l(9))];

    [X_1_measure, X_2_measure] = simulate_coupled_law(delta_t, T, X0_m, X0_m, Theta_SIG_l(1), Theta_SIG_l(2),Theta_SIG_l(3),Theta_SIG_l(4),Theta_SIG_l(5),Theta_SIG_l(6));
    [H1_l, H2_l, G_l] = particle_filter_coupled(Y_obs, X_1_measure, X_2_measure, delta_t, T, X0_pf, X0_pf, Theta_SIG_l(1),Theta_SIG_l(2),Theta_SIG_l(3),Theta_SIG_l(4),Theta_SIG_l(5),Theta_SIG_l(6),Theta_SIG_l(7),Theta_SIG_l(8),Theta_SIG_l(9));
    lG_l = G_l;
    l_pos_theta_l = l_posterior(Theta_l, lG_l);
    
    N_count_l = 0;
    for iter = 1:Nl(l+1)

        if mod(iter, 100) == 0
            fprintf('iter = %f\n', iter);
            disp(['AR = ', num2str(N_count_l/iter)]);
            disp(['H1 average = ', num2str(H1_sum/100)]);
            disp(['H2 average = ', num2str(H2_sum/100)]);
            disp(['current estimate = [', num2str(Theta_l(1)),  ', ' , num2str(Theta_l(2)), ', ',num2str(Theta_l(3)),  ', ', num2str(Theta_l(4)),  ', ' num2str(Theta_l(5)) ,  ', ', num2str(Theta_l(6)),  ', ' num2str(Theta_l(7)),  ', ', num2str(Theta_l(8)),  ', ' num2str(Theta_l(9)) ']']);
            H1_sum = 0;
            H2_sum = 0;
            toc;
            tic;
        end
        
        Theta_l_prime1 = Theta_l;
        Theta_l_prime = mvnrnd(Theta_l_prime1,Sigma_A*Sigma_A');
        Theta_l_SIG_prime = [Theta_l_prime(1), Theta_l_prime(2), exp(Theta_l_prime(3)),exp(Theta_l_prime(4)), exp(Theta_l_prime(5)),exp( Theta_l_prime(6)),exp(Theta_l_prime(7)),exp( Theta_l_prime(8)),exp(Theta_l_prime(9))];
        
        [X_1_measure_p, X_2_measure_p] = simulate_coupled_law(delta_t, T, X0_m, X0_m, Theta_l_SIG_prime(1),Theta_l_SIG_prime(2),Theta_l_SIG_prime(3),Theta_l_SIG_prime(4),Theta_l_SIG_prime(5),Theta_l_SIG_prime(6));
        [H1_lp, H2_lp, lG_lp] = particle_filter_coupled(Y_obs, X_1_measure_p, X_2_measure_p, delta_t, T, X0_pf, X0_pf,Theta_l_SIG_prime(1),Theta_l_SIG_prime(2),Theta_l_SIG_prime(3),Theta_l_SIG_prime(4),Theta_l_SIG_prime(5),Theta_l_SIG_prime(6),Theta_l_SIG_prime(7),Theta_l_SIG_prime(8),Theta_l_SIG_prime(9));
        l_pos_theta_l_prime = l_posterior(Theta_l_prime, lG_lp);
        alpha_l = min(0, l_pos_theta_l_prime - l_pos_theta_l);

        Ul = log(rand);
        if Ul < alpha_l

            Theta_l = Theta_l_prime;
            Theta_SIG_l = Theta_l_SIG_prime;
            Theta_trace{l+1, 1}(iter,:) = Theta_l_prime;
            X_1_measure = X_1_measure_p;
            X_2_measure = X_2_measure_p;
            lG_l = lG_lp;
            l_pos_theta_l = l_pos_theta_l_prime;
            H1_l = H1_lp;
            H2_l = H2_lp;
            H1_trace{l, 1}(iter,1) = H1_lp;
            H2_trace{l, 1}(iter,1) = H2_lp;
            N_count_l= N_count_l + 1;
            H1_sum = H1_sum + H1_lp;
            H2_sum = H2_sum + H2_lp;
            
        else
            Theta_trace{l+1, 1}(iter,:) = Theta_l; 
            [X_1_measure, X_2_measure] = simulate_coupled_law(delta_t, T, X0_m, X0_m, Theta_SIG_l(1), Theta_SIG_l(2),Theta_SIG_l(3),Theta_SIG_l(4),Theta_SIG_l(5),Theta_SIG_l(6));
            [H1_l, H2_l, lG_l] = particle_filter_coupled(Y_obs, X_1_measure, X_2_measure, delta_t, T, X0_pf, X0_pf,Theta_SIG_l(1),Theta_SIG_l(2),Theta_SIG_l(3),Theta_SIG_l(4),Theta_SIG_l(5),Theta_SIG_l(6),Theta_SIG_l(7),Theta_SIG_l(8),Theta_SIG_l(9));
            l_pos_theta_l = l_posterior(Theta_l, lG_l);
            H1_trace{l, 1}(iter,1) = H1_l;
            H2_trace{l, 1}(iter,1) = H2_l;
            H1_sum = H1_sum + H1_l;
            H2_sum = H2_sum + H2_l;
        end   
    end
        Aln(l+1,1) = N_count_l/ Nl(l+1);        
end
toc;

%% ==================== MLPMMH result analysis ====================
burnin = 1;

% Step 1: PMMH mean
E_0 = mean(Theta_trace{1,1}(burnin:end,:), 1);
Delta = zeros(LP - Lmin, 9);
ESS_fine = zeros(LP - Lmin, 1);
ESS_coarse = zeros(LP - Lmin, 1);

% Step 2: Calculate Delta、running mean、ESS for each level
for ll = 1:LP - Lmin
    H1_data = H1_trace{ll,1}(burnin:end,1);
    H2_data = H2_trace{ll,1}(burnin:end,1);
    Theta_data = Theta_trace{ll+1,1}(burnin:end,:);
    n_samples = size(Theta_data, 1);
    
    % Calculate the nonnormalized weights（用于 running mean）
    log_w1 = H1_data - max(H1_data);  
    log_w2 = H2_data - max(H2_data);
    raw_w1 = exp(log_w1);
    raw_w2 = exp(log_w2);
    
    %Normalized weights（用于 Delta 和 ESS）
    w1 = raw_w1 / sum(raw_w1);
    w2 = raw_w2 / sum(raw_w2);
    
    %  ESS
    ESS_fine(ll) = 1 / sum(w1.^2);
    ESS_coarse(ll) = 1 / sum(w2.^2);
    
    E_fine = sum(Theta_data .* w1, 1);
    E_coarse = sum(Theta_data .* w2, 1);
    
    %Delta
    Delta(ll,:) = E_fine - E_coarse;

    cumsum_w1 = cumsum(raw_w1);
    cumsum_w2 = cumsum(raw_w2);
    
    %Cumulative weighted theta
    cumsum_weighted_theta_1 = cumsum(Theta_data .* raw_w1, 1);
    cumsum_weighted_theta_2 = cumsum(Theta_data .* raw_w2, 1);
    
    % Running weighted mean = Cumulative weighted sum / Cumulative weight
    running_mean_fine = cumsum_weighted_theta_1 ./ cumsum_w1;
    running_mean_coarse = cumsum_weighted_theta_2 ./ cumsum_w2;
    running_delta = running_mean_fine - running_mean_coarse;
    % Running mean for plot
    
 
  
    % ============ figure ============
    % Plot1: Delta Running Mean
    figure('Name', ['Level ' num2str(ll + Lmin) ' - Delta Running Mean']);
    for i = 1:9
        subplot(3,3,i)
        plot(running_delta(:,i), 'b-', 'LineWidth', 1)
        yline(Delta(ll,i), 'r--', 'LineWidth', 1.5);  
        title(['Parameter ' num2str(i) ' - \Delta_{' num2str(ll + Lmin) '} Running Mean'])
        ylabel('\Delta')
        if i > 6 , xlabel('iterations'); end
    end
    
    
  %}
end
% Step 3: final estimate
final_estimate = E_0 + sum(Delta, 1);

% ============ output ============
fprintf('\n========== MLPMMH result ==========\n');
fprintf('PMMH_mean (E_0):\n');
fprintf('  I=%.4f, J=%.4f, log(c)=%.4f, log(λ)=%.4f, log(b_ext)=%.4f\n', E_0(1:5));
fprintf('  log(Γ)=%.4f, log(σ1)=%.4f, log(σ2)=%.4f, log(σ3)=%.4f\n', E_0(6:9));

for ll = 1:LP - Lmin
    fprintf('\nLevel %d Differential correction (Δ_%d):\n', ll + Lmin, ll + Lmin);
    fprintf('  [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n', Delta(ll,:));
    fprintf('  ESS_fine: %.1f / %d (%.1f%%)\n', ESS_fine(ll), Nl(ll+1)-burnin+1, 100*ESS_fine(ll)/(Nl(ll+1)-burnin+1));
    fprintf('  ESS_coarse: %.1f / %d (%.1f%%)\n', ESS_coarse(ll), Nl(ll+1)-burnin+1, 100*ESS_coarse(ll)/(Nl(ll+1)-burnin+1));
end

fprintf('\nFinal MLPMMH Estimate:\n');
fprintf('  I=%.4f, J=%.4f, log(c)=%.4f, log(λ)=%.4f, log(b_ext)=%.4f\n', final_estimate(1:5));
fprintf('  log(Γ)=%.4f, log(σ1)=%.4f, log(σ2)=%.4f, log(σ3)=%.4f\n', final_estimate(6:9));

fprintf('\nParameter after transformation:\n');
fprintf('  I=%.4f, J=%.4f, c=%.4f, λ=%.4f, b_ext=%.4f\n', ...
    final_estimate(1), final_estimate(2), exp(final_estimate(3)), exp(final_estimate(4)), exp(final_estimate(5)));
fprintf('  Γ=%.4f, σ1=%.4f, σ2=%.4f, σ3=%.4f\n', ...
    exp(final_estimate(6)), exp(final_estimate(7)), exp(final_estimate(8)), exp(final_estimate(9)));

% ============ PMMH convergence plot ============
figure('Name', 'PMMH Trace Plots');
param_names = {'I', 'J', 'log(c)', 'log(\lambda)', 'log(b_{ext})', ...
               'log(\Gamma)', 'log(\sigma_1)', 'log(\sigma_2)', 'log(\sigma_3)'};
for i = 1:9
    subplot(3,3,i)
    plot(Theta_trace{1,1}(:,i), 'b-', 'LineWidth', 0.5)
    xline(burnin, 'r--', 'LineWidth', 1);
    title(param_names{i})
    if i > 6, xlabel('iterations'); end
end
sgtitle('PMMH Trace Plots ')




% simulate laws
function X = simulate_law_model(delta_t, T, X0, I, J, c, lambda, b_ext, Gamma)
    particle_count = size(X0, 2);
    d = size(X0, 1);
    steps_count = round(T/delta_t);
    X = zeros(d,particle_count,steps_count+1);
    X(:,:,1) = X0;
    delta_W = sqrt(delta_t)*randn(d, particle_count, steps_count);
	delta_W_rep = zeros(size(delta_W,1),size(delta_W,1),size(delta_W,2), size(delta_W,3));
	for i = 1:size(delta_W,1)
		delta_W_rep(i,:,:,:) = delta_W;
	end
    for i = 1:steps_count
        X(:,:,i+1) = X(:,:,i) + drift(I, J, c, lambda, X(:,:,i), X(:,:,i)) * delta_t + ...
                          + squeeze(sum( diffusion(I, J, c, lambda, b_ext, Gamma, X(:,:,i), X(:,:,i), delta_t) .* delta_W_rep(:,:,:,i),2));
    end
end

% X is d*particles_count
function A = drift(Ii, Ji, ci, lambdai, X, X_law)
	I = Ii;
	J = Ji;
	V_rev = 1;
	c = ci;
	b = 0.8;
	a = 0.7;
	T_max = 1;
	lambda = lambdai;
	a_d = 1;
	a_r = 1;
	V_T = 2;
    particle_count = size(X, 2);
    d = size(X, 1);
    A = zeros(d,particle_count);
    A(1,:) = X(1,:) - X(1,:).^3/3 - X(2,:) + I - J*(X(1,:)-V_rev)*mean(X_law(3,:));
    A(2,:) = c*(X(1,:)+a-b*X(2,:));
	A(3,:) = a_r*T_max*(1-X(3,:))./(1+exp(-lambda*(X(1,:)-V_T))) - a_d*X(3,:);
end

% ideally it should return d*d*particles_count, but ignored for now...
function A = diffusion(Ii, Ji, ci, lambdai, b_exti, Gammai, X, X_law, delta_t)
	I = Ii;
	J = Ji;
	V_rev = 1;
	c = ci;
	b = 0.8;
	a = 0.7;
	T_max = 1;
	lambda = lambdai;
	a_d = 1;
	a_r = 1;
	V_T = 2;
	b_ext = b_exti;
	b_J = 0.2;
	Gamma = Gammai;
	Lambda = 0.5;
    d = size(X, 1);
    particle_count = size(X, 2);
	A = zeros(d,d,particle_count);
    A(1,1,:) = b_ext;
	A(1,3,:) = -b_J*(X(1,:)-V_rev)*mean(X_law(3,:));
    X_3 = (X(3,:)>0) .* (X(3,:)<1) .* X(3,:);
    A(3,2,:) = (X(3,:)>0) .* (X(3,:)<1) .* Gamma .* exp(-Lambda./(1-(2*X_3-1).^2)) .* sqrt(a_r*T_max*(1-X_3)./(1+exp(-lambda*(X(1,:)-V_T))));
    A(2,3,:) = delta_t;
end

function [X_1, X_2] = simulate_coupled_law(delta_t, T, X0_1, X0_2,  I, J, c, lambda, b_ext, Gamma)
    d = size(X0_1, 1);
    particle_count = size(X0_1, 2);  
    steps_count_2 = round(T/delta_t/2);      
    steps_count_1 = 2*steps_count_2;
    X_1 = zeros(d,particle_count, steps_count_1+1);
    X_2 = zeros(d,particle_count, steps_count_2+1);
    X_1(:,:,1) = X0_1;
    X_2(:,:,1) = X0_2;
    delta_W = sqrt(delta_t)*randn(d, particle_count, steps_count_1);
    delta_W_rep = zeros(size(delta_W,1),size(delta_W,1),size(delta_W,2), size(delta_W,3));
	for i = 1:size(delta_W,1)
		delta_W_rep(i,:,:,:) = delta_W;
	end

    for i = 1:steps_count_1
       X_1(:,:,i+1) = X_1(:,:,i) + drift(I, J, c, lambda,  X_1(:,:,i), X_1(:,:,i)) * delta_t + ...
                          + squeeze(sum(diffusion(I, J, c, lambda, b_ext, Gamma, X_1(:,:,i), X_1(:,:,i), delta_t) .* delta_W_rep(:,:,:,i),2));
     
        if mod(i,2) == 0
            j = ceil(i/2);
            X_2(:,:,j+1) = X_2(:,:,j) + drift(I, J, c, lambda, X_2(:,:,j), X_2(:,:,j)) *2* delta_t + ...
                          + squeeze(sum(diffusion(I, J, c, lambda, b_ext, Gamma, X_2(:,:,j), X_2(:,:,j), 2*delta_t) .* (delta_W_rep(:,:,:,i)+delta_W_rep(:,:,:,i-1)),2));    
        end
    end
end

function X0 = sample_X0(params, number)
    X0 = zeros(3, number);
    X0(1,:) = params(1) + sqrt(params(4))*randn(1, number);
    X0(2,:) = params(2) + sqrt(params(5))*randn(1, number);
    X0(3,:) = params(3) + sqrt(params(6))*randn(1, number);
end


 
function z = particle_filter(Y, X_measure, delta_t, T, X0, I, J, c, lambda, b_ext, Gamma, sigma_obs_1, sigma_obs_2, sigma_obs_3)
    d = size(X0, 1);
    particle_count = size(X0, 2);    
    steps_count = round(T/delta_t);
    X = zeros(d, particle_count, steps_count+1);
    X(:,:,1) = X0;
    delta_W = sqrt(delta_t)*randn(d, particle_count, steps_count);
	delta_W_rep = zeros(size(delta_W,1),size(delta_W,1),size(delta_W,2), size(delta_W,3));
	for i = 1:size(delta_W,1)
	    delta_W_rep(i,:,:,:) = delta_W;
    end

    log_w = zeros(particle_count, 1);
    k = 1;
    lGL_star = zeros(1,T);
    for i = 1:steps_count
        
        X(:,:,i+1) = X(:,:,i) + drift(I,J,c,lambda, X(:,:,i), X_measure(:,:,i)) * delta_t + ...
                          + squeeze(sum(diffusion( I, J, c, lambda, b_ext, Gamma, X(:,:,i), X_measure(:,:,i), delta_t) .* delta_W_rep(:,:,:,i),2));
      
        if i == round(k/delta_t)+1
            log_w = log_normpdf(X(:,:,i), Y(:,k), diag([sigma_obs_1, sigma_obs_2,sigma_obs_3]).^2);
            W = exp(log_w - max(log_w));
            lGL_star(1,k)= log(sum(W)) + max(log_w);
            W = W / sum(W);
            if 1/sum(W.^2) <= particle_count
                I = resampleSystematic(W);
                X = X(:,I,:);
            end
            k = k + 1;
        end
    end
    z = T * log(1/particle_count) + sum(lGL_star);
end



function [lwf,lwc,cz] = particle_filter_coupled(Y, X_1_measure, X_2_measure, delta_t, T, X0_1, X0_2, I, J, c, lambda, b_ext, Gamma, sigma_obs_1, sigma_obs_2, sigma_obs_3)
    
    d = size(X0_1, 1);
    particle_count = size(X0_1,2);    
    steps_count_2 = T/delta_t/2;       
    steps_count_1 = T/delta_t;
    X_1 = zeros(d, particle_count, steps_count_1+1);
    X_2 = zeros(d, particle_count, steps_count_2+1);
    X_1(:,:,1) = X0_1;
    X_2(:,:,1) = X0_2;

    delta_W = sqrt(delta_t)*randn(d, particle_count, steps_count_1);
	delta_W_rep = zeros(size(delta_W,1),size(delta_W,1),size(delta_W,2), size(delta_W,3));
	
    for i = 1:size(delta_W,1)
	    delta_W_rep(i,:,:,:) = delta_W;
    end

    log_w = zeros(particle_count, 1);
    k = 1;
    lGL_star = zeros(1,T);
    lwf = 0;
    lwc = 0;

    for i = 1:steps_count_1

        X_1(:,:,i+1) = X_1(:,:,i) + drift(I, J, c, lambda, X_1(:,:,i), X_1_measure(:,:,i)) * delta_t + ...
                          + squeeze(sum(diffusion( I, J, c, lambda, b_ext, Gamma, X_1(:,:,i), X_1_measure(:,:,i), delta_t) .* delta_W_rep(:,:,:,i),2));
     
        if mod(i,2) == 0
            j = ceil(i/2);
            X_2(:,:,j+1) = X_2(:,:,j) + drift(I, J, c, lambda,  X_2(:,:,j), X_2_measure(:,:,j)) *2* delta_t + ...
                          + squeeze(sum(diffusion(I, J, c, lambda, b_ext, Gamma, X_2(:,:,j), X_2_measure(:,:,j), 2*delta_t) .* (delta_W_rep(:,:,:,i)+delta_W_rep(:,:,:,i-1)),2));    
        end
        

        if mod(i, 1/delta_t)==0
            j = i/2;
            
            log_w_1 = log_normpdf(X_1(:,:,i+1), Y(round(i*delta_t)), diag([sigma_obs_1, sigma_obs_2,sigma_obs_3]).^2);
            log_w_2 = log_normpdf(X_2(:,:,j+1), Y(round(i*delta_t)), diag([sigma_obs_1, sigma_obs_2,sigma_obs_3]).^2);
            
            %log_w = log(1/2*(exp(log_w_1)+ exp(log_w_2)));
            log_w = max(log_w_1, log_w_2);
            
           % lwf = lwf + mean((log_w_1 - log_w));
           % lwc = lwc + mean((log_w_2 - log_w));
            W1 = exp(log_w_1 - max(log_w_1));
            log_Z1 = log(mean(W1)) + max(log_w_1);
            
            W2 = exp(log_w_2 - max(log_w_2));
            log_Z2 = log(mean(W2)) + max(log_w_2);
            
            W_coupled = exp(log_w - max(log_w));
            log_Z_coupled = log(mean(W_coupled)) + max(log_w);
            
            lwf = lwf + (log_Z1 - log_Z_coupled);
            lwc = lwc + (log_Z2 - log_Z_coupled);
            

            if lwf == inf
                 disp('inf lwf');
                cz = -inf;
                lwf = 0;
                lwc = 0;
                return
            end
            
            if lwc == inf
                 disp('inf lwc');
                cz = -inf;
                lwf = 0;
                lwc = 0;
                return
            end

            GL0 = exp(log_w - max(log_w));
            lGL_star(1,k)= log(sum(GL0)) + max(log_w);
           GLL = GL0 / sum(GL0);
        
            if isnan(sum(GLL)) 
                disp('ANNOYING NAN ERROR! GLL');
                cz = -inf;
                lwf = 0;
                lwc = 0;
                return
            end
            if  sum(GLL) == 0
                disp(' GLL = 0');
                cz = -inf;
                lwf = 0;
                lwc = 0;
                return
            end

           I = resampleSystematic( GLL);
           X_1 = X_1(:,I,:);
           X_2 = X_2(:,I,:);
           k = k + 1;
        end
    end
    %fprintf('lwf = = %f\n', lwf);
    %fprintf('lwc= = %f\n', lwc);
     cz = T * log(1/particle_count) + sum(lGL_star);
end



function lw = lG(y, x, Cov)
    k = size(x,2);
    lw  = -log(sqrt((2*pi)^k * det(Cov))) - 0.5*diag((y-x) * Cov^(-1) * (y-x)') ;
end

function lpos_p = l_posterior(Theta, lik)
    log_lik = lik;
    log_prior = lG(Theta(1),0.6,0.015) + lG(Theta(2),0.5,0.02) + lG(Theta(3),-2,0.02) + lG(Theta(4),-1.7,0.02) +lG(Theta(5),-1.3,0.02) + lG(Theta(6),-2,0.02) +lG(Theta(7),-0.2,0.02)+lG(Theta(8),-1,0.02)+lG(Theta(9),-2,0.02) ;
    lpos_p = log_lik + log_prior;
    
end

function  indx  = resampleSystematic(w)
    N = length(w);
    Q = cumsum(w);
    indx = zeros(1,N);
    T = linspace(0,1-1/N,N) + rand(1)/N;
    T(N+1) = 1;
    i=1;
    j=1;
    while (i<=N)
        if (T(i)<Q(j))
            indx(i)=j;
            i=i+1;
        else
            j=j+1;        
        end
    end
end

function a = log_normpdf(x,m,s)
    d = size(x,1);
    p_n = size(x,2);
    for i = 1:p_n
         a(i)  = -d/2*log(2*pi) -1/2*log( det(s)) - 0.5*(x(:,i)-m)' * inv(s) * (x(:,i)-m);
    end
    
end



