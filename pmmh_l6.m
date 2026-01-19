LP = 6;

job_id = getenv('SLURM_JOB_ID');
proc_id = getenv('SLURM_PROCID');
folder_read = '';
folder_write = sprintf('%s%s', job_id, '/');
results_filename = sprintf('%sL_%i_%s_%s.txt', folder_write, LP, job_id,proc_id);
rng_seed = sum(clock)*mod(str2num(job_id),10000)*(str2num(proc_id)+1);
rng(rng_seed);
format long

Y_obs = readmatrix(sprintf('%s%s', folder_read,'Y.txt'));
particle_count = 50;
T = 50;
d = 3;
params = [0, 0.5, 0.3, 0.4, 0.4, 0.05];
M = floor(0.3*2^(2*LP));
Nl =  floor(0.5 * 2^(2*LP)+2000);


X0_M = zeros(d,M);
X0_M = sample_X0(params, M);

X0_N = zeros(d,particle_count);
X0_N = sample_X0(params, particle_count);

Theta_trace = zeros(Nl,9);
delta = 2^(-LP);
Theta_A =  [0.7, 0.4, -2,-1.9,-1.2, -2, -0.1, -1, -2.2];
tic;

Theta_A_p = Theta_A;
Theta_SIG_p = [Theta_A_p(1), Theta_A_p(2), exp(Theta_A_p(3)), exp(Theta_A_p(4)),exp(Theta_A_p(5)),exp(Theta_A_p(6)),exp(Theta_A_p(7)),exp(Theta_A_p(8)),exp(Theta_A_p(9))];

X_measure = simulate_law_model(delta, T, X0_M, Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),Theta_SIG_p(5),Theta_SIG_p(6));
Z =  particle_filter(Y_obs, X_measure, delta, T, X0_N,Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),Theta_SIG_p(5),Theta_SIG_p(6),Theta_SIG_p(7),Theta_SIG_p(8),Theta_SIG_p(9));

lZ = Z;
l_pos_Theta_A_p = l_posterior(Theta_A_p, lZ);
Theta_trace(1,:) = Theta_A_p;

N_count_1 = 0;
N_count_last_1 = 0;
Sigma_A1 = 0.225*diag([1.25,1.1,1.1,1.15,1.2,1.15,1,1.1,1.2]);

for iter = 1:Nl(1)
 
    if mod(iter, 100) == 0
        fprintf('iter = %f\n', iter);
        disp(['current AC = ', num2str(N_count_1/(iter))]);
        disp(['current new AC = ', num2str((N_count_1 - N_count_last_1)/(mod(iter,50)+1))]);
        disp(['current estimate = [', num2str(Theta_A_p(1)),  ', ' , num2str(Theta_A_p(2)), ', ',num2str(Theta_A_p(3)),  ', ', num2str(Theta_A_p(4)),  ', ' num2str(Theta_A_p(5)) ,  ', ', num2str(Theta_A_p(6)),  ', ' num2str(Theta_A_p(7)) ,  ', ' num2str(Theta_A_p(8)),  ', ' num2str(Theta_A_p(9)) ']']);
        disp(['estimated posterior = ', num2str(l_pos_Theta_A_p)]);
    end
    if mod(iter, 100) == 0
        N_count_last_1 = N_count_1;
    end
    
    Theta_A_prime_1 = Theta_A_p;
    Theta_A_prime = mvnrnd(Theta_A_prime_1, Sigma_A1*Sigma_A1');
    Theta_SIG_prime = [ Theta_A_prime(1), Theta_A_prime(2), exp(Theta_A_prime(3)), exp(Theta_A_prime(4)),exp(Theta_A_prime(5)), exp(Theta_A_prime(6)),exp(Theta_A_prime(7)),exp(Theta_A_prime(8)),exp(Theta_A_prime(9))];

    X_measure_prime = simulate_law_model(delta, T, X0_M, Theta_SIG_prime(1), Theta_SIG_prime(2),Theta_SIG_prime(3),Theta_SIG_prime(4),Theta_SIG_prime(5),Theta_SIG_prime(6));
    Z_prime =  particle_filter(Y_obs, X_measure_prime, delta, T, X0_N,Theta_SIG_prime(1),Theta_SIG_prime(2),Theta_SIG_prime(3),Theta_SIG_prime(4),Theta_SIG_prime(5),Theta_SIG_prime(6),Theta_SIG_prime(7),Theta_SIG_prime(8),Theta_SIG_prime(9));
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
        Theta_trace(iter,:) = Theta_A_prime; 
        N_count_1 = N_count_1 + 1;        
    else
        Theta_trace(iter,:) = Theta_A_p; 
        
        lZ = particle_filter(Y_obs, X_measure, delta, T, X0_N, Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),Theta_SIG_p(5),Theta_SIG_p(6),Theta_SIG_p(7),Theta_SIG_p(8),Theta_SIG_p(9) );
        l_pos_Theta_A_p2 = l_posterior(Theta_A_p, lZ);  
        if true && (l_pos_Theta_A_p2 ~= -inf)
            l_pos_Theta_A_p = l_pos_Theta_A_p2;
        end       
        
    end 

end

Aln = N_count_1 / Nl(1);
burnin = 2000;
final_theta = mean(Theta_trace(burnin:end,:),1);
writematrix(final_theta, results_filename);


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

function X0 = sample_X0(params, number)
    X0 = zeros(3, number);
    X0(1,:) = params(1) + sqrt(params(4))*randn(1, number);
    X0(2,:) = params(2) + sqrt(params(5))*randn(1, number);
    X0(3,:) = params(3) + sqrt(params(6))*randn(1, number);
end

 
function z = particle_filter(Y, X_measure, delta_t, T, X0, I, J, c, lambda, b_ext, Gamma, sigma_obs_1, sigma_obs_2, sigma_obs_3)
    d = size(X0, 1);
    particle_count = size(X0,2);    
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

function lw = lG(y, x, Cov)
    k = size(x,2);
    lw  = -log(sqrt((2*pi)^k * det(Cov))) - 0.5*diag((y-x) * Cov^(-1) * (y-x)') ;
end

function lpos_p = l_posterior(Theta, lik)
    log_lik = lik;
    %log_prior = 0;
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



