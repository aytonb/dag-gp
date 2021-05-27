clear all;

kern_param1 = 1.25;
kern_param2 = 0.7;
q_mean_param1 = 1.3;
q_mean_param2 = 0.6;
q_chol_param1 = 1;
q_chol_param2 = 0.2;
q_chol_param3 = 1.2;
noise_param = -1;
parent_param1 = 1;
parent_param2 = 1.5;
y_par_mean1 = [1;0.3;0.6];
y_par_cov1 = [0,0,0;0,1,0.5;0,0.5,1];
y_par_mean2 = [2.2;0.5;0.7];
y_par_cov2 = [0.7,0.25,0.1;0.25,0.8,0.3;0.1,0.3,0.4];
y_par_mean3 = [1.7;0.4;0.8];
y_par_cov3 = [1.2,0.6,0.5;0.6,0.9,0.3;0.5,0.3,0.7];

% f = 0, 2, 5
ff_sq_dist = [0;0;0];

% u = 1, 3
uu_sq_dist = [0,4;4,0];

fu_sq_dist = [1,9;1,1;16,4];

% f pred = 6, 7
pred_ff_sq_dist1 = [0];
pred_ff_sq_dist2 = [0];

pred_fu_sq_dist1 = [25,9];
pred_fu_sq_dist2 = [36,16];


Kff = kern1(ff_sq_dist,kern_param1,kern_param2);
% Kuu has added jitter
Kuu = kern1(uu_sq_dist,kern_param1,kern_param2) + 1e-5 * eye(2);
Kfu = kern1(fu_sq_dist,kern_param1,kern_param2);


q_chol = [q_chol_param1,0;q_chol_param2,q_chol_param3];
q_cov = q_chol * q_chol';

q_mean = [q_mean_param1; q_mean_param2];
qf_mean = Kfu * (Kuu \ q_mean);

Kfu_post = (Kuu' \ Kfu')';
middle = q_cov - Kuu; 
KfuM = Kfu_post * middle;
 
Kff_post = KfuM * Kfu_post';
qf_cov = Kff + diag(Kff_post);

parent_params = [parent_param1,parent_param2];
log_lik = gauss_likelihood(qf_mean(1),qf_cov(1),noise_param,y_par_mean1,y_par_cov1,parent_params) + ...
    gauss_likelihood(qf_mean(2),qf_cov(2),noise_param,y_par_mean2,y_par_cov2,parent_params) + ...
    gauss_likelihood(qf_mean(3),qf_cov(3),noise_param,y_par_mean3,y_par_cov3,parent_params);
log_lik = log_lik - GaussianKL(q_mean,q_cov,zeros(2,1),Kuu)
 
derivs = zeros(10,1);

mean_derivs = [gauss_likelihood_mean_deriv(qf_mean(1),qf_cov(1),noise_param,y_par_mean1,y_par_cov1,parent_params); ...
               gauss_likelihood_mean_deriv(qf_mean(2),qf_cov(2),noise_param,y_par_mean2,y_par_cov2,parent_params); ...
               gauss_likelihood_mean_deriv(qf_mean(3),qf_cov(3),noise_param,y_par_mean3,y_par_cov3,parent_params)];

var_derivs = [gauss_likelihood_var_deriv(qf_mean(1),qf_cov(1),noise_param,y_par_mean1,y_par_cov1,parent_params); ...
              gauss_likelihood_var_deriv(qf_mean(2),qf_cov(2),noise_param,y_par_mean2,y_par_cov2,parent_params); ...
              gauss_likelihood_var_deriv(qf_mean(3),qf_cov(3),noise_param,y_par_mean3,y_par_cov3,parent_params)];
          
noise_derivs = [gauss_likelihood_noise_deriv(qf_mean(1),qf_cov(1),noise_param,y_par_mean1,y_par_cov1,parent_params); ...
                gauss_likelihood_noise_deriv(qf_mean(2),qf_cov(2),noise_param,y_par_mean2,y_par_cov2,parent_params); ...
                gauss_likelihood_noise_deriv(qf_mean(3),qf_cov(3),noise_param,y_par_mean3,y_par_cov3,parent_params)];

% temp_pred = Kuu' \ Kfu_pred';
% qf_mean_pred = temp_pred' * q_mean;
% qf_cov_pred = Kff_pred + temp_pred' * (q_cov - Kuu) * temp_pred;

%% Kernel derivs

Kff_kern_deriv1 = kern1d1(ff_sq_dist,kern_param1,kern_param2);
Kuu_kern_deriv1 = kern1d1(uu_sq_dist,kern_param1,kern_param2);
Kfu_kern_deriv1 = kern1d1(fu_sq_dist,kern_param1,kern_param2);
qf_mean_kern_deriv1 = Kfu_kern_deriv1 * (Kuu \ q_mean) - Kfu_post * Kuu_kern_deriv1 * (Kuu \ q_mean);
qf_cov_kern_deriv1_1 = Kff_kern_deriv1;
qf_cov_kern_deriv1_2 = diag((Kuu \ Kfu_kern_deriv1')' * (q_cov - Kuu) * Kfu_post') + diag(Kfu_post * (q_cov - Kuu) * (Kuu \ Kfu_kern_deriv1'));
qf_cov_kern_deriv1_3 = - diag(Kfu_post * Kuu_kern_deriv1 * (Kuu \ (q_cov - Kuu)) * Kfu_post') - diag(Kfu_post * (Kuu \ (q_cov - Kuu))' * Kuu_kern_deriv1 * Kfu_post') - ...
    diag(Kfu_post * Kuu_kern_deriv1 * Kfu_post');
qf_cov_kern_deriv1 = qf_cov_kern_deriv1_1 + qf_cov_kern_deriv1_2 + qf_cov_kern_deriv1_3;
KL_kern_deriv1 = 0.5* (-trace(Kuu \ (Kuu_kern_deriv1 * (Kuu \ q_cov))) - (Kuu \ q_mean)' * Kuu_kern_deriv1 * (Kuu \ q_mean) + trace(Kuu \ Kuu_kern_deriv1));
log_lik_kern_deriv1 = sum(mean_derivs .* qf_mean_kern_deriv1 + var_derivs .* qf_cov_kern_deriv1) - KL_kern_deriv1;
derivs(1) = log_lik_kern_deriv1;

Kff_kern_deriv2 = kern1d2(ff_sq_dist,kern_param1,kern_param2);
Kuu_kern_deriv2 = kern1d2(uu_sq_dist,kern_param1,kern_param2);
Kfu_kern_deriv2 = kern1d2(fu_sq_dist,kern_param1,kern_param2);
qf_mean_kern_deriv2 = Kfu_kern_deriv2 * (Kuu \ q_mean) - Kfu_post * Kuu_kern_deriv2 * (Kuu \ q_mean);
qf_cov_kern_deriv2_1 = Kff_kern_deriv2;
qf_cov_kern_deriv2_2 = diag((Kuu \ Kfu_kern_deriv2')' * (q_cov - Kuu) * Kfu_post') + diag(Kfu_post * (q_cov - Kuu) * (Kuu \ Kfu_kern_deriv2'));
qf_cov_kern_deriv2_3 = - diag(Kfu_post * Kuu_kern_deriv2 * (Kuu \ (q_cov - Kuu)) * Kfu_post') - diag(Kfu_post * (Kuu \ (q_cov - Kuu))' * Kuu_kern_deriv2 * Kfu_post') - ...
    diag(Kfu_post * Kuu_kern_deriv2 * Kfu_post');
qf_cov_kern_deriv2 = qf_cov_kern_deriv2_1 + qf_cov_kern_deriv2_2 + qf_cov_kern_deriv2_3;
KL_kern_deriv2 = 0.5* (-trace(Kuu \ (Kuu_kern_deriv2 * (Kuu \ q_cov))) - (Kuu \ q_mean)' * Kuu_kern_deriv2 * (Kuu \ q_mean) + trace(Kuu \ Kuu_kern_deriv2));
log_lik_kern_deriv2 = sum(mean_derivs .* qf_mean_kern_deriv2 + var_derivs .* qf_cov_kern_deriv2) - KL_kern_deriv2;
derivs(2) = log_lik_kern_deriv2;


%% noise derivs
log_lik_noisederiv = sum(noise_derivs);
derivs(3) = log_lik_noisederiv;


%% q_mean derivs
q_mean_meanderiv1 = [1;0];
qf_mean_meanderiv1 = Kfu_post * q_mean_meanderiv1;
KL_meanderiv1 = q_mean_meanderiv1' * (Kuu \ q_mean);
log_lik_meanderiv1 = sum(mean_derivs .* qf_mean_meanderiv1) - KL_meanderiv1;
derivs(4) = log_lik_meanderiv1;

q_mean_meanderiv2 = [0;1];
qf_mean_meanderiv2 = Kfu_post * q_mean_meanderiv2;
KL_meanderiv2 = q_mean_meanderiv2' * (Kuu \ q_mean);
log_lik_meanderiv2 = sum(mean_derivs .* qf_mean_meanderiv2) - KL_meanderiv2;
derivs(5) = log_lik_meanderiv2;


%% q_chol derivs
qL_deriv11 = [1,0;0,0];
q_cov_cholderiv11 = qL_deriv11 * q_chol' + q_chol * qL_deriv11';
KL_cov_chol_deriv11 = 0.5 * (trace(Kuu \ q_cov_cholderiv11) - trace(q_cov \ q_cov_cholderiv11));
qf_cov_cholderiv11 = diag(Kfu_post * q_cov_cholderiv11 * Kfu_post');
log_lik_cholderiv11 = sum(var_derivs .* qf_cov_cholderiv11) - KL_cov_chol_deriv11;
derivs(6) = log_lik_cholderiv11;

qL_deriv21 = [0,0;1,0];
q_cov_cholderiv21 = qL_deriv21 * q_chol' + q_chol * qL_deriv21';
KL_cov_chol_deriv21 = 0.5 * (trace(Kuu \ q_cov_cholderiv21) - trace(q_cov \ q_cov_cholderiv21));
qf_cov_cholderiv21 = diag(Kfu_post * q_cov_cholderiv21 * Kfu_post');
log_lik_cholderiv21 = sum(var_derivs .* qf_cov_cholderiv21) - KL_cov_chol_deriv21;
derivs(7) = log_lik_cholderiv21;

qL_deriv22 = [0,0;0,1];
q_cov_cholderiv22 = qL_deriv22 * q_chol' + q_chol * qL_deriv22';
KL_cov_chol_deriv22 = 0.5 * (trace(Kuu \ q_cov_cholderiv22) - trace(q_cov \ q_cov_cholderiv22));
qf_cov_cholderiv22 = diag(Kfu_post * q_cov_cholderiv22 * Kfu_post');
log_lik_cholderiv22 = sum(var_derivs .* qf_cov_cholderiv22) - KL_cov_chol_deriv22;
derivs(8) = log_lik_cholderiv22;


%% parent derivs

parent_derivs1 = [gauss_likelihood_parent_deriv1(qf_mean(1),qf_cov(1),noise_param,y_par_mean1,y_par_cov1,parent_params); ...
                  gauss_likelihood_parent_deriv1(qf_mean(2),qf_cov(2),noise_param,y_par_mean2,y_par_cov2,parent_params); ...
                  gauss_likelihood_parent_deriv1(qf_mean(3),qf_cov(3),noise_param,y_par_mean3,y_par_cov3,parent_params)];
              
log_lik_parentderiv1 = sum(parent_derivs1);
derivs(9) = log_lik_parentderiv1;

parent_derivs2 = [gauss_likelihood_parent_deriv2(qf_mean(1),qf_cov(1),noise_param,y_par_mean1,y_par_cov1,parent_params); ...
                  gauss_likelihood_parent_deriv2(qf_mean(2),qf_cov(2),noise_param,y_par_mean2,y_par_cov2,parent_params); ...
                  gauss_likelihood_parent_deriv2(qf_mean(3),qf_cov(3),noise_param,y_par_mean3,y_par_cov3,parent_params)];
              
log_lik_parentderiv2= sum(parent_derivs2);
derivs(10) = log_lik_parentderiv2;

derivs

%% Prediction

pred_Kff1 = kern1(pred_ff_sq_dist1,kern_param1,kern_param2);
pred_Kfu1 = kern1(pred_fu_sq_dist1,kern_param1,kern_param2);
pred_Kff2 = kern1(pred_ff_sq_dist2,kern_param1,kern_param2);
pred_Kfu2 = kern1(pred_fu_sq_dist2,kern_param1,kern_param2);

pred_Kfu_post1 = (Kuu' \ pred_Kfu1')';
pred_Kfu_post2 = (Kuu' \ pred_Kfu2')';

pred_qf_mean1 = pred_Kfu_post1 * q_mean
pred_qf_mean2 = pred_Kfu_post2 * q_mean

pred_KfuM1 = pred_Kfu_post1 * middle;
pred_KfuM2 = pred_Kfu_post2 * middle;
 
pred_qf_cov1 = pred_Kff1 + pred_KfuM1 * pred_Kfu_post1'
pred_qf_cov2 = pred_Kff2 + pred_KfuM2 * pred_Kfu_post2'


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