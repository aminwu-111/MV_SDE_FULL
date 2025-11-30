
clear;
clc;
format long

T = 50;
particle_count = 50;
law_particles_count = 50;
Y_obs = readmatrix(['d3_obs_T_50.txt']);
params = [0, 0.5, 0.3, 0.4, 0.4, 0.05];
d = 3;
Lmin = 4;
LP = 4;

hl = zeros(LP - Lmin + 1, 1);
hl(1) = 0.5*2^(-Lmin);
for l = 1:LP- Lmin
    hl(l+1) = 2^(-l- Lmin);
end

%number of iterations for each level
Nl = 1000;
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
    Theta_trace{k, 1} = zeros(Nl(k),7);
    Theta_traceN{k,1} = zeros(Nl(k),7);
    ML_Theta_trace{k, 1} = zeros(Nl(k),7);
end

for i = 1:LP - Lmin
    Theta_trace_1{i,1} = zeros(Nl(i+1),7);
    Theta_trace_2{i,1} = zeros(Nl(i+1),7);

    Theta_trace_1N{i,1} = zeros(Nl(i+1),7);
    Theta_trace_2N{i,1} = zeros(Nl(i+1),7);

    H1_trace{i,1} = zeros(Nl(i+1),1);
    H2_trace{i,1} = zeros(Nl(i+1),1);
end


delta = 0.5*2^(-LP);
%we are trying to estimate I J c lambda b_ext, Gamma and sig_obs
%Theta_A = [0.5, 1.1, -2,-1,-1, -2, -0.3];
Theta_A =  [0.8, 1.1, -2,-1,-1, -2, -0.3];
tic;
Theta_A_p = Theta_A;
Theta_SIG_p = [Theta_A_p(1), Theta_A_p(2), exp(Theta_A_p(3)), exp(Theta_A_p(4)),exp(Theta_A_p(5)),exp(Theta_A_p(6)),exp(Theta_A_p(7))];
X0 = zeros(d,particle_count);
X0 = sample_X0(params, particle_count);

X_measure = simulate_law_model(delta, T, X0, Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),Theta_SIG_p(5),Theta_SIG_p(6));
Z =  particle_filter(Y_obs, X_measure, delta, T, X0,Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),Theta_SIG_p(5),Theta_SIG_p(6),Theta_SIG_p(7));

lZ = Z;
l_pos_Theta_A_p = l_posterior(Theta_A_p, lZ);
Theta_trace{1,1}(1,:) = Theta_A_p;

N_count_1 = 0;
N_count_last_1 = 0;
%Sigma_A1 = 0.3*diag([0.85,0.85,0.65,0.85,0.8,0.8,0.6]);
Sigma_A1 = 0.3*diag([0.95,0.95,0.75,0.95,1.1,1.1,0.6]);
%Sigma_A1 = 1.2*diag([1]);
for iter = 1:Nl(1)
 
    if mod(iter, 100) == 0
        fprintf('iter = %f\n', iter);
        disp(['current AC = ', num2str(N_count_1/(iter))]);
        disp(['current new AC = ', num2str((N_count_1 - N_count_last_1)/(mod(iter,50)+1))]);
        disp(['current estimate = [', num2str(Theta_A_p(1)),  ', ' , num2str(Theta_A_p(2)), ', ',num2str(Theta_A_p(3)),  ', ', num2str(Theta_A_p(4)),  ', ' num2str(Theta_A_p(5)) ,  ', ', num2str(Theta_A_p(6)),  ', ' num2str(Theta_A_p(7)) ']']);
        disp(['estimated posterior = ', num2str(l_pos_Theta_A_p)]);
    end
    if mod(iter, 100) == 0
        N_count_last_1 = N_count_1;
    end
    
    Theta_A_prime_1 = Theta_A_p;
    Theta_A_prime = mvnrnd(Theta_A_prime_1, Sigma_A1*Sigma_A1');
    Theta_SIG_prime = [ Theta_A_prime(1), Theta_A_prime(2), exp(Theta_A_prime(3)), exp(Theta_A_prime(4)),exp(Theta_A_prime(5)), exp(Theta_A_prime(6)),exp(Theta_A_prime(7))];

    X_measure_prime = simulate_law_model(delta, T, X0, Theta_SIG_prime(1), Theta_SIG_prime(2),Theta_SIG_prime(3),Theta_SIG_prime(4),Theta_SIG_prime(5),Theta_SIG_prime(6));
    Z_prime =  particle_filter(Y_obs, X_measure_prime, delta, T, X0,Theta_SIG_prime(1),Theta_SIG_prime(2),Theta_SIG_prime(3),Theta_SIG_prime(4),Theta_SIG_prime(5),Theta_SIG_prime(6),Theta_SIG_prime(7));
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
        
        lZ = particle_filter(Y_obs, X_measure, delta, T, X0, Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),Theta_SIG_p(5),Theta_SIG_p(6),Theta_SIG_p(7) );
        l_pos_Theta_A_p2 = l_posterior(Theta_A_p, lZ);  
        if true && (l_pos_Theta_A_p2 ~= -inf)
            l_pos_Theta_A_p = l_pos_Theta_A_p2;
        end       
        
    end 

end

Aln = N_count_1 / Nl(1);
burnin = 1;
figure
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,1), 'r--');
title('I')
hold off

figure
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,2), 'r--');
title('J')
hold off

figure
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,3), 'r--');
title('log(c)')
hold off

figure
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,4), 'r--');
title('log(\lambda)')
hold off

figure
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,5), 'r--');
title('log(b\_ext)')
hold off

figure
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,6), 'r--');
title('log(\Gamma)')
hold off

figure
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,7), 'r--');
title('log(\tau)')
hold off


%{

figure
subplot(5,1,1)
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,1), 'r--');
title('I')
subplot(5,1,2)
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,2), 'r--');
title('J')
subplot(5,1,3)
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,3), 'r--');
title('log(c)')
subplot(5,1,4)
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,4), 'r--');
title('log(\lambda)')
subplot(5,1,5)
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,5), 'r--');
title('log(\sigma_{obs})')

%}
%{
Theta_iters = Theta_trace{1,1};
burnin = 1;
niter = 5000;
desired_height = 0.12;
figure_distance = 400;
f = figure;
f.Position = f.Position+[0 -figure_distance 0 figure_distance];

ax = subplot(4,1,1);
plot(burnin:3:niter,Theta_iters(burnin:3:end,1), 'r-',LineWidth=1);
title('I');
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');

ax = subplot(4,1,2);
plot(burnin:3:niter,Theta_iters(burnin:3:end,2), 'r-',LineWidth=1);
title('J');
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');


ax = subplot(4,1,3);
plot(burnin:3:niter,Theta_iters(burnin:3:end,3), 'r-',LineWidth=1);
title('log(c)')
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');

ax = subplot(4,1,4);
plot(burnin:3:niter,Theta_iters(burnin:3:end,4), 'r-',LineWidth=1);
title('log(\lambda)')
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');


niter = 5000;
desired_height = 0.12;
figure_distance = 400;
f = figure;
f.Position = f.Position+[0 -figure_distance 0 figure_distance];

ax = subplot(3,1,1);
plot(burnin:3:niter,Theta_iters(burnin:3:end,1), 'r-',LineWidth=1);
title('log(b_{ext})');
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');

ax = subplot(3,1,2);
plot(burnin:3:niter,Theta_iters(burnin:3:end,2), 'r-',LineWidth=1);
title('log(\Gamma)');
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');


ax = subplot(3,1,3);
plot(burnin:3:niter,Theta_iters(burnin:3:end,3), 'r-',LineWidth=1);
title('log(\tau) ')
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');
%}
%}
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

 
function z = particle_filter(Y, X_measure, delta_t, T, X0, I, J, c, lambda, b_ext, Gamma, sigma_obs)
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
            log_w = log_normpdf(X(:,:,i), Y(:,k), sigma_obs^2*eye(3));
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
    log_prior = lG(Theta(1),0.6,0.01) + lG(Theta(2),1,0.01) + lG(Theta(3),-2,0.01) + lG(Theta(4),-1,0.01) +lG(Theta(5),-1,0.01) + lG(Theta(6),-2,0.01) +lG(Theta(7),-0.8,0.01) ;
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



