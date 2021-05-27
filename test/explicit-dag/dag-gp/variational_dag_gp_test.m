clear all;

kern1_param1 = 0.75;
kern1_param2 = 1.2;
kern2_param1 = 1.25;
kern2_param2 = 0.7;
kern3_param1 = 0.8;
kern3_param2 = 0.9;

noise1_param = -1.2;
noise2_param = -1;
noise3_param = -1.6;

parent_param1 = 1;
parent_param2 = 1.5;

q_mean1_param1 = 0.3;
q_mean1_param2 = 0.5;
q_mean2_param1 = 1.3;
q_mean2_param2 = 0.6;
q_mean3_param1 = 0.9;
q_mean3_param2 = 0.5;

q_chol1_param1 = 0.7;
q_chol1_param2 = 0.2;
q_chol1_param3 = 0.8;
q_chol2_param1 = 1;
q_chol2_param2 = 0.2;
q_chol2_param3 = 1.2;
q_chol3_param1 = 0.4;
q_chol3_param2 = 0.1;
q_chol3_param3 = 0.7;


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


Kff1 = kern1(ff_sq_dist,kern1_param1,kern1_param2);
Kff2 = kern1(ff_sq_dist,kern2_param1,kern2_param2);
Kff3 = kern1(ff_sq_dist,kern3_param1,kern3_param2);
% Kuu has added jitter
Kuu1 = kern1(uu_sq_dist,kern1_param1,kern1_param2) + 1e-5 * eye(2);
Kuu2 = kern1(uu_sq_dist,kern2_param1,kern2_param2) + 1e-5 * eye(2);
Kuu3 = kern1(uu_sq_dist,kern3_param1,kern3_param2) + 1e-5 * eye(2);

Kfu1 = kern1(fu_sq_dist,kern1_param1,kern1_param2);
Kfu2 = kern1(fu_sq_dist,kern2_param1,kern2_param2);
Kfu3 = kern1(fu_sq_dist,kern3_param1,kern3_param2);

q_chol1 = [q_chol1_param1,0;q_chol1_param2,q_chol1_param3];
q_cov1 = q_chol1 * q_chol1';
q_chol2 = [q_chol2_param1,0;q_chol2_param2,q_chol2_param3];
q_cov2 = q_chol2 * q_chol2';
q_chol3 = [q_chol3_param1,0;q_chol3_param2,q_chol3_param3];
q_cov3 = q_chol3 * q_chol3';

q_mean1 = [q_mean1_param1; q_mean1_param2];
qf_mean1 = Kfu1 * (Kuu1 \ q_mean1);
q_mean2 = [q_mean2_param1; q_mean2_param2];
qf_mean2 = Kfu2 * (Kuu2 \ q_mean2);
q_mean3 = [q_mean3_param1; q_mean3_param2];
qf_mean3 = Kfu3 * (Kuu3 \ q_mean3);

Kfu_post1 = (Kuu1' \ Kfu1')';
middle1 = q_cov1 - Kuu1; 
KfuM1 = Kfu_post1 * middle1;
Kfu_post2 = (Kuu2' \ Kfu2')';
middle2 = q_cov2 - Kuu2; 
KfuM2 = Kfu_post2 * middle2;
Kfu_post3 = (Kuu3' \ Kfu3')';
middle3 = q_cov3 - Kuu3; 
KfuM3 = Kfu_post3 * middle3;
 
Kff_post1 = KfuM1 * Kfu_post1';
qf_cov1 = Kff1 + diag(Kff_post1);
Kff_post2 = KfuM2 * Kfu_post2';
qf_cov2 = Kff2 + diag(Kff_post2);
Kff_post3 = KfuM3 * Kfu_post3';
qf_cov3 = Kff3 + diag(Kff_post3);

a = [1,0,0;
     -parent_param1,1,-parent_param2;
     0,0,1];
a = inv(a);
a = [a,a];

%% Intermediates (conditioned on local y)
Kff_int1 = diag([Kff1(1),Kff2(1),Kff3(1)]);
Kfefe_int1 = [Kff_int1, zeros(3,3);
              zeros(3,3), diag([exp(noise1_param),exp(noise2_param),exp(noise3_param)])];
Kfefey_int1 = [Kfefe_int1, Kfefe_int1* a(1,:)';
               a(1,:) * Kfefe_int1, a(1,:) * Kfefe_int1 * a(1,:)'];
Kfefegy_int1 = Kfefey_int1(1:6,1:6) - Kfefey_int1(1:6,7) * (Kfefey_int1(7,7) \ Kfefey_int1(7,1:6));
meanfefegy_int1 = Kfefey_int1(1:6,7) * (Kfefey_int1(7,7) \ [1]);

Kff_int2 = diag([Kff1(2),Kff2(2),Kff3(2)]);
Kfefe_int2 = [Kff_int2, zeros(3,3);
              zeros(3,3), diag([exp(noise1_param),exp(noise2_param),exp(noise3_param)])];
Kfefey_int2 = [Kfefe_int2, Kfefe_int2 * a(1:2,:)';
               a(1:2,:) * Kfefe_int2, a(1:2,:) * Kfefe_int2 * a(1:2,:)'];
Kfefegy_int2 = Kfefey_int2(1:6,1:6) - Kfefey_int2(1:6,7:8) * (Kfefey_int2(7:8,7:8) \ Kfefey_int2(7:8,1:6));
meanfefegy_int2 = Kfefey_int2(1:6,7:8) * (Kfefey_int2(7:8,7:8) \ [2;1.5]);

Kff_int3 = diag([Kff1(3),Kff2(3),Kff3(3)]);
Kfefe_int3 = [Kff_int3, zeros(3,3);
              zeros(3,3), diag([exp(noise1_param),exp(noise2_param),exp(noise3_param)])];
Kfefey_int3 = [Kfefe_int3, Kfefe_int3 * a(2:3,:)';
               a(2:3,:) * Kfefe_int3, a(2:3,:) * Kfefe_int3 * a(2:3,:)'];
Kfefegy_int3 = Kfefey_int3(1:6,1:6) - Kfefey_int3(1:6,7:8) * (Kfefey_int3(7:8,7:8) \ Kfefey_int3(7:8,1:6));
meanfefegy_int3 = Kfefey_int3(1:6,7:8) * (Kfefey_int3(7:8,7:8) \ [4;2.5]);


%% Conditioned on all y
Kffgy1 = diag([qf_cov1(1),qf_cov2(1),qf_cov3(1)]);
meanfgy1 = [qf_mean1(1);qf_mean2(1);qf_mean3(1)];

Kffgy2 = diag([qf_cov1(2),qf_cov2(2),qf_cov3(2)]);
meanfgy2 = [qf_mean1(2);qf_mean2(2);qf_mean3(2)];

Kffgy3 = diag([qf_cov1(3),qf_cov2(3),qf_cov3(3)]);
meanfgy3 = [qf_mean1(3);qf_mean2(3);qf_mean3(3)];


%% Reconstruction
temp1 = Kfefegy_int1([1:3],[1:3]) \ Kfefegy_int1([1:3],[4:6]);
Keegy1 = Kfefegy_int1([4:6],[4:6]) + temp1' * (Kffgy1 - Kfefegy_int1([1:3],[1:3])) * temp1;
Kfegy1 = Kffgy1 * temp1;
Kfefegy1 = [Kffgy1, Kfegy1;
            Kfegy1', Keegy1];
meanegy1 = meanfefegy_int1(4:6) + temp1' * (meanfgy1 - meanfefegy_int1(1:3));
meanfegy1 = [meanfgy1; meanegy1];
meany1 = a * meanfegy1
Kyy1 = a * Kfefegy1 * a'

temp2 = Kfefegy_int2([1:3],[1:3]) \ Kfefegy_int2([1:3],[4:6]);
Keegy2 = Kfefegy_int2([4:6],[4:6]) + temp2' * (Kffgy2 - Kfefegy_int2([1:3],[1:3])) * temp2;
Kfegy2 = Kffgy2 * temp2;
Kfefegy2 = [Kffgy2, Kfegy2;
            Kfegy2', Keegy2];
meanegy2 = meanfefegy_int2(4:6) + temp2' * (meanfgy2 - meanfefegy_int2(1:3));
meanfegy2 = [meanfgy2; meanegy2];
meany2 = a * meanfegy2
Kyy2 = a * Kfefegy2 * a'

temp3 = Kfefegy_int3([1:3],[1:3]) \ Kfefegy_int3([1:3],[4:6]);
Keegy3 = Kfefegy_int3([4:6],[4:6]) + temp3' * (Kffgy3 - Kfefegy_int3([1:3],[1:3])) * temp3;
Kfegy3 = Kffgy3 * temp3;
Kfefegy3 = [Kffgy3, Kfegy3;
            Kfegy3', Keegy3];
meanegy3 = meanfefegy_int3(4:6) + temp3' * (meanfgy3 - meanfefegy_int3(1:3));
meanfegy3 = [meanfgy3; meanegy3];
meany3 = a * meanfegy3
Kyy3 = a * Kfefegy3 * a'


% parent_params = [parent_param1,parent_param2];
% log_lik = gauss_likelihood(qf_mean(1),qf_cov(1),noise_param,y_par_mean1,y_par_cov1,parent_params) + ...
%     gauss_likelihood(qf_mean(2),qf_cov(2),noise_param,y_par_mean2,y_par_cov2,parent_params) + ...
%     gauss_likelihood(qf_mean(3),qf_cov(3),noise_param,y_par_mean3,y_par_cov3,parent_params);
% log_lik = log_lik - GaussianKL(q_mean,q_cov,zeros(2,1),Kuu)
%  
% derivs = zeros(10,1);
% 
% mean_derivs = [gauss_likelihood_mean_deriv(qf_mean(1),qf_cov(1),noise_param,y_par_mean1,y_par_cov1,parent_params); ...
%                gauss_likelihood_mean_deriv(qf_mean(2),qf_cov(2),noise_param,y_par_mean2,y_par_cov2,parent_params); ...
%                gauss_likelihood_mean_deriv(qf_mean(3),qf_cov(3),noise_param,y_par_mean3,y_par_cov3,parent_params)];
% 
% var_derivs = [gauss_likelihood_var_deriv(qf_mean(1),qf_cov(1),noise_param,y_par_mean1,y_par_cov1,parent_params); ...
%               gauss_likelihood_var_deriv(qf_mean(2),qf_cov(2),noise_param,y_par_mean2,y_par_cov2,parent_params); ...
%               gauss_likelihood_var_deriv(qf_mean(3),qf_cov(3),noise_param,y_par_mean3,y_par_cov3,parent_params)];
%           
% noise_derivs = [gauss_likelihood_noise_deriv(qf_mean(1),qf_cov(1),noise_param,y_par_mean1,y_par_cov1,parent_params); ...
%                 gauss_likelihood_noise_deriv(qf_mean(2),qf_cov(2),noise_param,y_par_mean2,y_par_cov2,parent_params); ...
%                 gauss_likelihood_noise_deriv(qf_mean(3),qf_cov(3),noise_param,y_par_mean3,y_par_cov3,parent_params)];

% temp_pred = Kuu' \ Kfu_pred';
% qf_mean_pred = temp_pred' * q_mean;
% qf_cov_pred = Kff_pred + temp_pred' * (q_cov - Kuu) * temp_pred;





%% Prediction

% pred_Kff1 = kern1(pred_ff_sq_dist1,kern_param1,kern_param2);
% pred_Kfu1 = kern1(pred_fu_sq_dist1,kern_param1,kern_param2);
% pred_Kff2 = kern1(pred_ff_sq_dist2,kern_param1,kern_param2);
% pred_Kfu2 = kern1(pred_fu_sq_dist2,kern_param1,kern_param2);
% 
% pred_Kfu_post1 = (Kuu' \ pred_Kfu1')';
% pred_Kfu_post2 = (Kuu' \ pred_Kfu2')';
% 
% pred_qf_mean1 = pred_Kfu_post1 * q_mean
% pred_qf_mean2 = pred_Kfu_post2 * q_mean
% 
% pred_KfuM1 = pred_Kfu_post1 * middle;
% pred_KfuM2 = pred_Kfu_post2 * middle;
%  
% pred_qf_cov1 = pred_Kff1 + pred_KfuM1 * pred_Kfu_post1'
% pred_qf_cov2 = pred_Kff2 + pred_KfuM2 * pred_Kfu_post2'


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