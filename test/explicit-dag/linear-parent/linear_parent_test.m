clear all;

kern_param1 = 1.25;
kern_param2 = 0.7;
noise_param = -1;
parent_param1 = 1;
parent_param2 = 1.5;
y_par_mean = [1;0.3;0.6;2.2;0.5;0.7;1.7;0.4;0.8];
y_par_cov = [0,0,0,0,0,0,0,0,0;
              0,1,0.5,0.4,0.3,0.1,0.2,0.2,0.3;
              0,0.5,1,0.1,0.14,0.2,0.2,0.3,0.2;
              0,0.4,0.1,0.7,0.25,0.1,0.05,0.2,0.3;
              0,0.3,0.14,0.25,0.8,0.3,0.03,0.1,0.1;
              0,0.1,0.2,0.1,0.3,0.4,0.13,0.1,0.24;
              0,0.2,0.2,0.05,0.03,0.13,1.2,0.6,0.16;
              0,0.2,0.3,0.2,0.1,0.1,0.6,0.9,0.3;
              0,0.3,0.2,0.3,0.1,0.24,0.16,0.3,0.4];
test_expression = false;

% f = 0, 2, 5
ff_sq_dist = [0,4,25;
              4,0,9;
              25,9,0];


% f pred = 6, 7
pred_ff_sq_dist1 = [0];
pred_ff_sq_dist2 = [0];

pred_fu_sq_dist1 = [25,9];
pred_fu_sq_dist2 = [36,16];


Kff = kern1(ff_sq_dist,kern_param1,kern_param2) + 1e-5 * eye(3) + exp(noise_param) * eye(3);
A = [eye(3), -parent_param1*eye(3), -parent_param2*eye(3)];

y_pred = A*y_par_mean;
y_pred_cov = A * y_par_cov * A';
log_lik = -1/2 * y_pred' * (Kff \ y_pred) - 1/2 * log(det(Kff)) -1/2 * trace(Kff \ y_pred_cov)

if test_expression
    empirical_LL = 0;
    for i = 1:1e7
        sample = mvnrnd(y_par_mean(2:9),y_par_cov(2:9,2:9));
        sample = [1;sample'];
        sample = A*sample;
        empirical_LL = empirical_LL - 1/2 * sample' * (Kff \ sample) - 1/2 * log(det(Kff)); 
    end
    empirical_LL = empirical_LL/1e6   
end


%% Kernel derivs

Kff_kern_deriv1 = kern1d1(ff_sq_dist,kern_param1,kern_param2);
log_lik_kern_deriv1 = 1/2 * y_pred' * (Kff \ (Kff \ Kff_kern_deriv1)') * y_pred - 1/2 * trace(Kff \ Kff_kern_deriv1) + 1/2 * trace(Kff \ (Kff \ Kff_kern_deriv1)' * y_pred_cov);
derivs(1) = log_lik_kern_deriv1;

Kff_kern_deriv2 = kern1d2(ff_sq_dist,kern_param1,kern_param2);
log_lik_kern_deriv2 = 1/2 * y_pred' * (Kff \ (Kff \ Kff_kern_deriv2)') * y_pred - 1/2 * trace(Kff \ Kff_kern_deriv2) + 1/2 * trace(Kff \ (Kff \ Kff_kern_deriv2)' * y_pred_cov);
derivs(2) = log_lik_kern_deriv2;

 
%% noise derivs
Kff_noisederiv = exp(noise_param) * eye(3);
log_lik_noisederiv = 1/2 * y_pred' * (Kff \ (Kff \ Kff_noisederiv)') * y_pred - 1/2 * trace(Kff \ Kff_noisederiv) + 1/2 * trace(Kff \ (Kff \ Kff_noisederiv)' * y_pred_cov);
derivs(3) = log_lik_noisederiv;


%% parent derivs

A_parentderiv1 = [zeros(3,3), -eye(3), zeros(3,3)];
y_parentderiv1 = [-y_par_mean(4); -y_par_mean(5); -y_par_mean(6)];
y_cov_parentderiv1 = A_parentderiv1 * y_par_cov * A';
y_cov_parentderiv1 = y_cov_parentderiv1 + y_cov_parentderiv1';
log_lik_parent_deriv1 = -1/2 * 2 * y_parentderiv1' * (Kff \ y_pred) - 1/2 * trace(Kff \ y_cov_parentderiv1);
derivs(4) = log_lik_parent_deriv1;

A_parentderiv2 = [zeros(3,3), zeros(3,3), -eye(3)];
y_parentderiv2 = [-y_par_mean(7); -y_par_mean(8); -y_par_mean(9)];
y_cov_parentderiv2 = A_parentderiv2 * y_par_cov * A';
y_cov_parentderiv2 = y_cov_parentderiv2 + y_cov_parentderiv2';
log_lik_parent_deriv2 = -1/2 * 2 * y_parentderiv2' * (Kff \ y_pred) - 1/2 * trace(Kff \ y_cov_parentderiv2);
derivs(5) = log_lik_parent_deriv2;

derivs
 

function out = kern1(in,param1,param2)

out = exp(param1).*exp(-in./exp(param2));

end

function out = kern1d1(in,param1,param2)
out = exp(param1).*exp(-in./exp(param2));
end

function out = kern1d2(in,param1,param2)
out = exp(param1).*in/exp(param2).*exp(-in./exp(param2)); 
end

function out = kern2(in,param1,param2)

out = exp(param1).*exp(-in./exp(param2));

end


function LL = gauss_likelihood(f_mean,f_var,noise_param,y_mean,y_cov,parent_params)
eps_var = exp(noise_param);
A = [-1,parent_params];
LL = -(y_mean(1) - f_mean - parent_params(1)*y_mean(2) - parent_params(2)*y_mean(3))^2 / (2 * eps_var) - 1/2 * log(eps_var) - f_var/(2*eps_var) - (A * y_cov * A')/(2*eps_var);
end

function d = gauss_likelihood_mean_deriv(f_mean,f_var,noise_param,y_mean,y_cov,parent_params)
eps_var = exp(noise_param);
A = [-1,parent_params];
d = -2*(y_mean(1) - f_mean - parent_params(1)*y_mean(2) - parent_params(2)*y_mean(3))*-1 / (2 * eps_var);
end

function d = gauss_likelihood_var_deriv(f_mean,f_var,noise_param,y_mean,y_cov,parent_params)
eps_var = exp(noise_param);
A = [-1,parent_params];
d= -1/(2*eps_var);
end

function d = gauss_likelihood_noise_deriv(f_mean,f_var,noise_param,y_mean,y_cov,parent_params)
eps_var = exp(noise_param);
A = [-1,parent_params];
d = (y_mean(1) - f_mean - parent_params(1)*y_mean(2) - parent_params(2)*y_mean(3))^2 / (2 * eps_var^2) * exp(noise_param) - 1/(2 * eps_var) * exp(noise_param) + f_var/(2*eps_var^2) * exp(noise_param) + (A * y_cov * A')/(2*eps_var^2) * exp(noise_param);
end

function d = gauss_likelihood_parent_deriv1(f_mean,f_var,noise_param,y_mean,y_cov,parent_params)
eps_var = exp(noise_param);
A = [-1,parent_params];
Ad = [0,1,0];
d = -2*(y_mean(1) - f_mean - parent_params(1)*y_mean(2) - parent_params(2)*y_mean(3)) * (-y_mean(2)) / (2 * eps_var) - (Ad * y_cov * A')/(2*eps_var) - (A * y_cov * Ad')/(2*eps_var);
end

function d = gauss_likelihood_parent_deriv2(f_mean,f_var,noise_param,y_mean,y_cov,parent_params)
eps_var = exp(noise_param);
A = [-1,parent_params];
Ad = [0,0,1];
d = -2*(y_mean(1) - f_mean - parent_params(1)*y_mean(2) - parent_params(2)*y_mean(3)) * (-y_mean(3)) / (2 * eps_var) - (Ad * y_cov * A')/(2*eps_var) - (A * y_cov * Ad')/(2*eps_var);
end

function out = GaussianKL(mu0,sigma0,mu1,sigma1)
out = 0.5 * (trace(sigma1\sigma0) + (mu1-mu0)' * (sigma1 \ (mu1-mu0)) + log(det(sigma1)) - log(det(sigma0)));
end