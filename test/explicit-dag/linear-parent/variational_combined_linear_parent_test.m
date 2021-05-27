clear all;

kern_param11 = 1.25;
kern_param21 = 0.7;
kern_param12 = 0.3;
kern_param22 = 1.2;
q_mean_param11 = 1.3;
q_mean_param21 = 0.6;
q_mean_param12 = 0.8;
q_mean_param22 = 0.5;
q_chol_param11 = 1;
q_chol_param21 = 0.2;
q_chol_param31 = 1.2;
q_chol_param12 = 0.6;
q_chol_param22 = 1.4;
q_chol_param32 = 0.3;
parent_param111 = 1;
parent_param211 = 2;
parent_param311 = 3;
parent_param121 = -1;
parent_param221 = -2;
parent_param321 = -3;
parent_param112 = 0;
parent_param212 = 1;
parent_param312 = 1.5;
parent_param122 = 3;
parent_param222 = 0;
parent_param322 = 2.3;

% Quadrature sets at every location
quad_prob1 = [2,2,0,0.4;         
              2,0,1,0.01;
              2,1,1,0.02;
              2,2,1,0.07;
              2,0,2,0.1;
              2,1,2,0.2;
              2,2,2,0.2];
quad1 = quad_prob1(:,1:3);
weight1 = quad_prob1(:,4);
quad_prob2 = [2,0,0,0.04;
              0,1,0,0.02;
              1,1,0,0.02;
              0,2,0,0.16;
              1,2,0,0.16;  
              2,0,1,0.02;
              2,1,1,0.03;
              0,2,1,0.025;
              1,2,1,0.025;
              2,0,2,0.15;
              2,1,2,0.25;
              2,2,2,0.1];
quad2 = quad_prob2(:,1:3);
weight2 = quad_prob2(:,4);
quad_prob3 = [2,0,0,0.04;
              0,1,0,0.02;
              1,1,0,0.02;
              0,2,0,0.16;
              1,2,0,0.16;  
              2,0,1,0.02;
              2,1,1,0.03;
              0,2,1,0.025;
              1,2,1,0.025;
              2,0,2,0.15;
              2,1,2,0.25;
              2,2,2,0.1];
quad3 = quad_prob3(:,1:3);
weight3 = quad_prob3(:,4);

% f = 0, 2, 5
ff_sq_dist = [0;0;0];

% u = 1, 3
uu_sq_dist = [0,4;4,0];

fu_sq_dist = [1,9;1,1;16,4];

% % f pred = 6, 7
% pred_ff_sq_dist1 = [0];
% pred_ff_sq_dist2 = [0];
% 
% pred_fu_sq_dist1 = [25,9];
% pred_fu_sq_dist2 = [36,16];


Kff1 = kern1(ff_sq_dist,kern_param11,kern_param21);
Kff2 = kern1(ff_sq_dist,kern_param12,kern_param22);
% Kuu has added jitter
Kuu1 = kern1(uu_sq_dist,kern_param11,kern_param21) + 1e-5 * eye(2);
Kuu2 = kern1(uu_sq_dist,kern_param12,kern_param22) + 1e-5 * eye(2);
Kfu1 = kern1(fu_sq_dist,kern_param11,kern_param21);
Kfu2 = kern1(fu_sq_dist,kern_param12,kern_param22);


q_chol1 = [q_chol_param11,0;q_chol_param21,q_chol_param31];
q_chol2 = [q_chol_param12,0;q_chol_param22,q_chol_param32];
q_cov1 = q_chol1 * q_chol1';
q_cov2 = q_chol2 * q_chol2';

q_mean1 = [q_mean_param11; q_mean_param21];
q_mean2 = [q_mean_param12; q_mean_param22];
qf_mean1 = Kfu1 * (Kuu1 \ q_mean1);
qf_mean2 = Kfu2 * (Kuu2 \ q_mean2);

Kfu_post1 = (Kuu1' \ Kfu1')';
Kfu_post2 = (Kuu2' \ Kfu2')';
middle1 = q_cov1 - Kuu1; 
middle2 = q_cov2 - Kuu2; 
KfuM1 = Kfu_post1 * middle1;
KfuM2 = Kfu_post2 * middle2;
 
Kff_post1 = KfuM1 * Kfu_post1';
Kff_post2 = KfuM2 * Kfu_post2';
qf_cov1 = Kff1 + diag(Kff_post1);
qf_cov2 = Kff2 + diag(Kff_post2);

parent_params11 = [parent_param111,parent_param211,parent_param311];
parent_params21 = [parent_param121,parent_param221,parent_param321];
parent_params12 = [parent_param112,parent_param212,parent_param312];
parent_params22 = [parent_param122,parent_param222,parent_param322];


log_lik = ternary_likelihood(qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22) + ...
    ternary_likelihood(qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22) + ...
    ternary_likelihood(qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22);
log_lik = log_lik - GaussianKL(q_mean1,q_cov1,zeros(2,1),Kuu1) - GaussianKL(q_mean2,q_cov2,zeros(2,1),Kuu2)
 
derivs = zeros(26,1);

mean_derivs1 = [ternary_likelihood_mean_deriv(1,qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22); ...
                ternary_likelihood_mean_deriv(1,qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22); ...
                ternary_likelihood_mean_deriv(1,qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22)];
           
mean_derivs2 = [ternary_likelihood_mean_deriv(2,qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22); ...
                ternary_likelihood_mean_deriv(2,qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22); ...
                ternary_likelihood_mean_deriv(2,qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22)];
            
var_derivs1 = [ternary_likelihood_var_deriv(1,qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22); ...
               ternary_likelihood_var_deriv(1,qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22); ...
               ternary_likelihood_var_deriv(1,qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22)];
           
var_derivs2 = [ternary_likelihood_var_deriv(2,qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22); ...
               ternary_likelihood_var_deriv(2,qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22); ...
               ternary_likelihood_var_deriv(2,qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22)];
           
% 
% % temp_pred = Kuu' \ Kfu_pred';
% % qf_mean_pred = temp_pred' * q_mean;
% % qf_cov_pred = Kff_pred + temp_pred' * (q_cov - Kuu) * temp_pred;

%% Kernel derivs

Kff_kern_deriv11 = kern1d1(ff_sq_dist,kern_param11,kern_param21);
Kuu_kern_deriv11 = kern1d1(uu_sq_dist,kern_param11,kern_param21);
Kfu_kern_deriv11 = kern1d1(fu_sq_dist,kern_param11,kern_param21);
qf_mean_kern_deriv11 = Kfu_kern_deriv11 * (Kuu1 \ q_mean1) - Kfu_post1 * Kuu_kern_deriv11 * (Kuu1 \ q_mean1);
qf_cov_kern_deriv11_1 = Kff_kern_deriv11;
qf_cov_kern_deriv11_2 = diag((Kuu1 \ Kfu_kern_deriv11')' * (q_cov1 - Kuu1) * Kfu_post1') + diag(Kfu_post1 * (q_cov1 - Kuu1) * (Kuu1 \ Kfu_kern_deriv11'));
qf_cov_kern_deriv11_3 = - diag(Kfu_post1 * Kuu_kern_deriv11 * (Kuu1 \ (q_cov1 - Kuu1)) * Kfu_post1') - diag(Kfu_post1 * (Kuu1 \ (q_cov1 - Kuu1))' * Kuu_kern_deriv11 * Kfu_post1') - ...
    diag(Kfu_post1 * Kuu_kern_deriv11 * Kfu_post1');
qf_cov_kern_deriv11 = qf_cov_kern_deriv11_1 + qf_cov_kern_deriv11_2 + qf_cov_kern_deriv11_3;
KL_kern_deriv11 = 0.5* (-trace(Kuu1 \ (Kuu_kern_deriv11 * (Kuu1 \ q_cov1))) - (Kuu1 \ q_mean1)' * Kuu_kern_deriv11 * (Kuu1 \ q_mean1) + trace(Kuu1 \ Kuu_kern_deriv11));
log_lik_kern_deriv11 = sum(mean_derivs1 .* qf_mean_kern_deriv11 + var_derivs1 .* qf_cov_kern_deriv11) - KL_kern_deriv11;
derivs(1) = log_lik_kern_deriv11;

Kff_kern_deriv21 = kern1d2(ff_sq_dist,kern_param11,kern_param21);
Kuu_kern_deriv21 = kern1d2(uu_sq_dist,kern_param11,kern_param21);
Kfu_kern_deriv21 = kern1d2(fu_sq_dist,kern_param11,kern_param21);
qf_mean_kern_deriv21 = Kfu_kern_deriv21 * (Kuu1 \ q_mean1) - Kfu_post1 * Kuu_kern_deriv21 * (Kuu1 \ q_mean1);
qf_cov_kern_deriv21_1 = Kff_kern_deriv21;
qf_cov_kern_deriv21_2 = diag((Kuu1 \ Kfu_kern_deriv21')' * (q_cov1 - Kuu1) * Kfu_post1') + diag(Kfu_post1 * (q_cov1 - Kuu1) * (Kuu1 \ Kfu_kern_deriv21'));
qf_cov_kern_deriv21_3 = - diag(Kfu_post1 * Kuu_kern_deriv21 * (Kuu1 \ (q_cov1 - Kuu1)) * Kfu_post1') - diag(Kfu_post1 * (Kuu1 \ (q_cov1 - Kuu1))' * Kuu_kern_deriv21 * Kfu_post1') - ...
    diag(Kfu_post1 * Kuu_kern_deriv21 * Kfu_post1');
qf_cov_kern_deriv21 = qf_cov_kern_deriv21_1 + qf_cov_kern_deriv21_2 + qf_cov_kern_deriv21_3;
KL_kern_deriv21 = 0.5* (-trace(Kuu1 \ (Kuu_kern_deriv21 * (Kuu1 \ q_cov1))) - (Kuu1 \ q_mean1)' * Kuu_kern_deriv21 * (Kuu1 \ q_mean1) + trace(Kuu1 \ Kuu_kern_deriv21));
log_lik_kern_deriv21 = sum(mean_derivs1 .* qf_mean_kern_deriv21 + var_derivs1 .* qf_cov_kern_deriv21) - KL_kern_deriv21;
derivs(2) = log_lik_kern_deriv21;

Kff_kern_deriv12 = kern1d1(ff_sq_dist,kern_param12,kern_param22);
Kuu_kern_deriv12 = kern1d1(uu_sq_dist,kern_param12,kern_param22);
Kfu_kern_deriv12 = kern1d1(fu_sq_dist,kern_param12,kern_param22);
qf_mean_kern_deriv12 = Kfu_kern_deriv12 * (Kuu2 \ q_mean2) - Kfu_post2 * Kuu_kern_deriv12 * (Kuu2 \ q_mean2);
qf_cov_kern_deriv12_1 = Kff_kern_deriv12;
qf_cov_kern_deriv12_2 = diag((Kuu2 \ Kfu_kern_deriv12')' * (q_cov2 - Kuu2) * Kfu_post2') + diag(Kfu_post2 * (q_cov2 - Kuu2) * (Kuu2 \ Kfu_kern_deriv12'));
qf_cov_kern_deriv12_3 = - diag(Kfu_post2 * Kuu_kern_deriv12 * (Kuu2 \ (q_cov2 - Kuu2)) * Kfu_post2') - diag(Kfu_post2 * (Kuu2 \ (q_cov2 - Kuu2))' * Kuu_kern_deriv12 * Kfu_post2') - ...
    diag(Kfu_post2 * Kuu_kern_deriv12 * Kfu_post2');
qf_cov_kern_deriv12 = qf_cov_kern_deriv12_1 + qf_cov_kern_deriv12_2 + qf_cov_kern_deriv12_3;
KL_kern_deriv12 = 0.5* (-trace(Kuu2 \ (Kuu_kern_deriv12 * (Kuu2 \ q_cov2))) - (Kuu2 \ q_mean2)' * Kuu_kern_deriv12 * (Kuu2 \ q_mean2) + trace(Kuu2 \ Kuu_kern_deriv12));
log_lik_kern_deriv12 = sum(mean_derivs2 .* qf_mean_kern_deriv12 + var_derivs2 .* qf_cov_kern_deriv12) - KL_kern_deriv12;
derivs(3) = log_lik_kern_deriv12;

Kff_kern_deriv22 = kern1d2(ff_sq_dist,kern_param12,kern_param22);
Kuu_kern_deriv22 = kern1d2(uu_sq_dist,kern_param12,kern_param22);
Kfu_kern_deriv22 = kern1d2(fu_sq_dist,kern_param12,kern_param22);
qf_mean_kern_deriv22 = Kfu_kern_deriv22 * (Kuu2 \ q_mean2) - Kfu_post2 * Kuu_kern_deriv22 * (Kuu2 \ q_mean2);
qf_cov_kern_deriv22_1 = Kff_kern_deriv22;
qf_cov_kern_deriv22_2 = diag((Kuu2 \ Kfu_kern_deriv22')' * (q_cov2 - Kuu2) * Kfu_post2') + diag(Kfu_post2 * (q_cov2 - Kuu2) * (Kuu2 \ Kfu_kern_deriv22'));
qf_cov_kern_deriv22_3 = - diag(Kfu_post2 * Kuu_kern_deriv22 * (Kuu2 \ (q_cov2 - Kuu2)) * Kfu_post2') - diag(Kfu_post2 * (Kuu2 \ (q_cov2 - Kuu2))' * Kuu_kern_deriv22 * Kfu_post2') - ...
    diag(Kfu_post2 * Kuu_kern_deriv22 * Kfu_post2');
qf_cov_kern_deriv22 = qf_cov_kern_deriv22_1 + qf_cov_kern_deriv22_2 + qf_cov_kern_deriv22_3;
KL_kern_deriv22 = 0.5* (-trace(Kuu2 \ (Kuu_kern_deriv22 * (Kuu2 \ q_cov2))) - (Kuu2 \ q_mean2)' * Kuu_kern_deriv22 * (Kuu2 \ q_mean2) + trace(Kuu2 \ Kuu_kern_deriv22));
log_lik_kern_deriv22 = sum(mean_derivs2 .* qf_mean_kern_deriv22 + var_derivs2 .* qf_cov_kern_deriv22) - KL_kern_deriv22;
derivs(4) = log_lik_kern_deriv22;


%% q_mean derivs
q_mean_meanderiv11 = [1;0];
qf_mean_meanderiv11 = Kfu_post1 * q_mean_meanderiv11;
KL_meanderiv11 = q_mean_meanderiv11' * (Kuu1 \ q_mean1);
log_lik_meanderiv11 = sum(mean_derivs1 .* qf_mean_meanderiv11) - KL_meanderiv11;
derivs(5) = log_lik_meanderiv11;

q_mean_meanderiv21 = [0;1];
qf_mean_meanderiv21 = Kfu_post1 * q_mean_meanderiv21;
KL_meanderiv21 = q_mean_meanderiv21' * (Kuu1 \ q_mean1);
log_lik_meanderiv21 = sum(mean_derivs1 .* qf_mean_meanderiv21) - KL_meanderiv21;
derivs(6) = log_lik_meanderiv21;

q_mean_meanderiv12 = [1;0];
qf_mean_meanderiv12 = Kfu_post2 * q_mean_meanderiv12;
KL_meanderiv12 = q_mean_meanderiv12' * (Kuu2 \ q_mean2);
log_lik_meanderiv12 = sum(mean_derivs2 .* qf_mean_meanderiv12) - KL_meanderiv12;
derivs(7) = log_lik_meanderiv12;

q_mean_meanderiv22 = [0;1];
qf_mean_meanderiv22 = Kfu_post2 * q_mean_meanderiv22;
KL_meanderiv22 = q_mean_meanderiv22' * (Kuu2 \ q_mean2);
log_lik_meanderiv22 = sum(mean_derivs2 .* qf_mean_meanderiv22) - KL_meanderiv22;
derivs(8) = log_lik_meanderiv22;


%% q_chol derivs
qL_deriv111 = [1,0;0,0];
q_cov_cholderiv111 = qL_deriv111 * q_chol1' + q_chol1 * qL_deriv111';
KL_cov_chol_deriv111 = 0.5 * (trace(Kuu1 \ q_cov_cholderiv111) - trace(q_cov1 \ q_cov_cholderiv111));
qf_cov_cholderiv111 = diag(Kfu_post1 * q_cov_cholderiv111 * Kfu_post1');
log_lik_cholderiv111 = sum(var_derivs1 .* qf_cov_cholderiv111) - KL_cov_chol_deriv111;
derivs(9) = log_lik_cholderiv111;

qL_deriv211 = [0,0;1,0];
q_cov_cholderiv211 = qL_deriv211 * q_chol1' + q_chol1 * qL_deriv211';
KL_cov_chol_deriv211 = 0.5 * (trace(Kuu1 \ q_cov_cholderiv211) - trace(q_cov1 \ q_cov_cholderiv211));
qf_cov_cholderiv211 = diag(Kfu_post1 * q_cov_cholderiv211 * Kfu_post1');
log_lik_cholderiv211 = sum(var_derivs1 .* qf_cov_cholderiv211) - KL_cov_chol_deriv211;
derivs(10) = log_lik_cholderiv211;

qL_deriv221 = [0,0;0,1];
q_cov_cholderiv221 = qL_deriv221 * q_chol1' + q_chol1 * qL_deriv221';
KL_cov_chol_deriv221 = 0.5 * (trace(Kuu1 \ q_cov_cholderiv221) - trace(q_cov1 \ q_cov_cholderiv221));
qf_cov_cholderiv221 = diag(Kfu_post1 * q_cov_cholderiv221 * Kfu_post1');
log_lik_cholderiv221 = sum(var_derivs1 .* qf_cov_cholderiv221) - KL_cov_chol_deriv221;
derivs(11) = log_lik_cholderiv221;

qL_deriv112 = [1,0;0,0];
q_cov_cholderiv112 = qL_deriv112 * q_chol2' + q_chol2 * qL_deriv112';
KL_cov_chol_deriv112 = 0.5 * (trace(Kuu2 \ q_cov_cholderiv112) - trace(q_cov2 \ q_cov_cholderiv112));
qf_cov_cholderiv112 = diag(Kfu_post2 * q_cov_cholderiv112 * Kfu_post2');
log_lik_cholderiv112 = sum(var_derivs2 .* qf_cov_cholderiv112) - KL_cov_chol_deriv112;
derivs(12) = log_lik_cholderiv112;

qL_deriv212 = [0,0;1,0];
q_cov_cholderiv212 = qL_deriv212 * q_chol2' + q_chol2 * qL_deriv212';
KL_cov_chol_deriv212 = 0.5 * (trace(Kuu2 \ q_cov_cholderiv212) - trace(q_cov2 \ q_cov_cholderiv212));
qf_cov_cholderiv212 = diag(Kfu_post2 * q_cov_cholderiv212 * Kfu_post2');
log_lik_cholderiv212 = sum(var_derivs2 .* qf_cov_cholderiv212) - KL_cov_chol_deriv212;
derivs(13) = log_lik_cholderiv212;

qL_deriv222 = [0,0;0,1];
q_cov_cholderiv222 = qL_deriv222 * q_chol2' + q_chol2 * qL_deriv222';
KL_cov_chol_deriv222 = 0.5 * (trace(Kuu2 \ q_cov_cholderiv222) - trace(q_cov2 \ q_cov_cholderiv222));
qf_cov_cholderiv222 = diag(Kfu_post2 * q_cov_cholderiv222 * Kfu_post2');
log_lik_cholderiv222 = sum(var_derivs2 .* qf_cov_cholderiv222) - KL_cov_chol_deriv222;
derivs(14) = log_lik_cholderiv222;

%% parent derivs

parent_derivs111 = [ternary_likelihood_parent_deriv(1,1,1,qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(1,1,1,qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(1,1,1,qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22)];
              
log_lik_parentderiv111 = sum(parent_derivs111);
derivs(15) = log_lik_parentderiv111;

parent_derivs121 = [ternary_likelihood_parent_deriv(1,1,2,qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(1,1,2,qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(1,1,2,qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22)];
              
log_lik_parentderiv121 = sum(parent_derivs121);
derivs(16) = log_lik_parentderiv121;

parent_derivs131 = [ternary_likelihood_parent_deriv(1,1,3,qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(1,1,3,qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(1,1,3,qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22)];
              
log_lik_parentderiv131 = sum(parent_derivs131);
derivs(17) = log_lik_parentderiv131;

parent_derivs211 = [ternary_likelihood_parent_deriv(1,2,1,qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(1,2,1,qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(1,2,1,qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22)];
              
log_lik_parentderiv211 = sum(parent_derivs211);
derivs(18) = log_lik_parentderiv211;

parent_derivs221 = [ternary_likelihood_parent_deriv(1,2,2,qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(1,2,2,qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(1,2,2,qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22)];
              
log_lik_parentderiv221 = sum(parent_derivs221);
derivs(19) = log_lik_parentderiv221;

parent_derivs231 = [ternary_likelihood_parent_deriv(1,2,3,qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(1,2,3,qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(1,2,3,qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22)];
              
log_lik_parentderiv231 = sum(parent_derivs231);
derivs(20) = log_lik_parentderiv231;

parent_derivs112 = [ternary_likelihood_parent_deriv(2,1,1,qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(2,1,1,qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(2,1,1,qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22)];
              
log_lik_parentderiv112 = sum(parent_derivs112);
derivs(21) = log_lik_parentderiv112;

parent_derivs122 = [ternary_likelihood_parent_deriv(2,1,2,qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(2,1,2,qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(2,1,2,qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22)];
              
log_lik_parentderiv122 = sum(parent_derivs122);
derivs(22) = log_lik_parentderiv122;

parent_derivs132 = [ternary_likelihood_parent_deriv(2,1,3,qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(2,1,3,qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(2,1,3,qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22)];
              
log_lik_parentderiv132 = sum(parent_derivs132);
derivs(23) = log_lik_parentderiv132;

parent_derivs212 = [ternary_likelihood_parent_deriv(2,2,1,qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(2,2,1,qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(2,2,1,qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22)];
              
log_lik_parentderiv212 = sum(parent_derivs212);
derivs(24) = log_lik_parentderiv212;

parent_derivs222 = [ternary_likelihood_parent_deriv(2,2,2,qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(2,2,2,qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(2,2,2,qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22)];
              
log_lik_parentderiv222 = sum(parent_derivs222);
derivs(25) = log_lik_parentderiv222;

parent_derivs232 = [ternary_likelihood_parent_deriv(2,2,3,qf_mean1(1),qf_mean2(1),qf_cov1(1),qf_cov2(1),quad1,weight1,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(2,2,3,qf_mean1(2),qf_mean2(2),qf_cov1(2),qf_cov2(2),quad2,weight2,parent_params11,parent_params21,parent_params12,parent_params22); ...
                    ternary_likelihood_parent_deriv(2,2,3,qf_mean1(3),qf_mean2(3),qf_cov1(3),qf_cov2(3),quad3,weight3,parent_params11,parent_params21,parent_params12,parent_params22)];
              
log_lik_parentderiv232 = sum(parent_derivs232);
derivs(26) = log_lik_parentderiv232;


derivs


% %% Prediction
% 
% pred_Kff1 = kern1(pred_ff_sq_dist1,kern_param11,kern_param21);
% pred_Kfu1 = kern1(pred_fu_sq_dist1,kern_param11,kern_param21);
% pred_Kff2 = kern1(pred_ff_sq_dist2,kern_param11,kern_param21);
% pred_Kfu2 = kern1(pred_fu_sq_dist2,kern_param11,kern_param21);
% 
% pred_Kfu_post1 = (Kuu1' \ pred_Kfu1')';
% pred_Kfu_post2 = (Kuu1' \ pred_Kfu2')';
% 
% pred_qf_mean1 = pred_Kfu_post1 * q_mean1
% pred_qf_mean2 = pred_Kfu_post2 * q_mean1
% 
% pred_KfuM1 = pred_Kfu_post1 * middle1;
% pred_KfuM2 = pred_Kfu_post2 * middle1;
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


function LL = ternary_likelihood(f_mean1,f_mean2,f_var1,f_var2,y_quad,y_weight,parent_params11,parent_params21,parent_params12,parent_params22)
hermite_quad = [-2.350604973674492222834; -1.335849074013696949715; -0.4360774119276165086792; 0.4360774119276165086792; 1.335849074013696949715; 2.350604973674492222834];
hermite_weight = [0.0025557844020562465; 0.08861574604191454; 0.40882846955602925; 0.40882846955602925; 0.08861574604191454; 0.0025557844020562465];
f_quad = [];
f_weight = [];
hermite_combined = [];
n_y_quad = length(y_weight);
for i = 1:6
    for j = 1:6
        f_quad = [f_quad; f_mean1 + sqrt(2*f_var1)*hermite_quad(i), f_mean2 + sqrt(2*f_var2)*hermite_quad(j)];
        f_weight = [f_weight; hermite_weight(i)*hermite_weight(j)];
        hermite_combined = [hermite_combined; hermite_quad(i), hermite_quad(j)];
    end
end

LL = 0;
for i = 1:36
    f_weight_i = f_weight(i);
    for j = 1:n_y_quad
        y = y_quad(j,1);
        par1 = y_quad(j,2);
        par2 = y_quad(j,3);
        f_effective_1 = f_quad(i,1);
        f_effective_2 = f_quad(i,2);
        if par1 == 0
            f_effective_1 = f_effective_1 + parent_params11(1);
            f_effective_2 = f_effective_2 + parent_params12(1);
        elseif par1 == 1
            f_effective_1 = f_effective_1 + parent_params11(2);
            f_effective_2 = f_effective_2 + parent_params12(2);
        else
            f_effective_1 = f_effective_1 + parent_params11(3);
            f_effective_2 = f_effective_2 + parent_params12(3);
        end
        if par2 == 0
            f_effective_1 = f_effective_1 + parent_params21(1);
            f_effective_2 = f_effective_2 + parent_params22(1);
        elseif par2 == 1
            f_effective_1 = f_effective_1 + parent_params21(2);
            f_effective_2 = f_effective_2 + parent_params22(2);
        else
            f_effective_1 = f_effective_1 + parent_params21(3);
            f_effective_2 = f_effective_2 + parent_params22(3);
        end
        if y == 0
            LL = LL + f_weight_i * y_weight(j) * (f_effective_1 - log(exp(f_effective_1) + exp(f_effective_2) + 1));
        elseif y == 1
            LL = LL + f_weight_i * y_weight(j) * (f_effective_2 - log(exp(f_effective_1) + exp(f_effective_2) + 1));
        else
            LL = LL - f_weight_i * y_weight(j) * (log(exp(f_effective_1) + exp(f_effective_2) + 1));
        end
    end
end

end

function d = ternary_likelihood_mean_deriv(deriv_index,f_mean1,f_mean2,f_var1,f_var2,y_quad,y_weight,parent_params11,parent_params21,parent_params12,parent_params22)
hermite_quad = [-2.350604973674492222834; -1.335849074013696949715; -0.4360774119276165086792; 0.4360774119276165086792; 1.335849074013696949715; 2.350604973674492222834];
hermite_weight = [0.0025557844020562465; 0.08861574604191454; 0.40882846955602925; 0.40882846955602925; 0.08861574604191454; 0.0025557844020562465];
f_quad = [];
f_weight = [];
hermite_combined = [];
n_y_quad = length(y_weight);
for i = 1:6
    for j = 1:6
        f_quad = [f_quad; f_mean1 + sqrt(2*f_var1)*hermite_quad(i), f_mean2 + sqrt(2*f_var2)*hermite_quad(j)];
        f_weight = [f_weight; hermite_weight(i)*hermite_weight(j)];
        hermite_combined = [hermite_combined; hermite_quad(i), hermite_quad(j)];
    end
end

d = 0;
for i = 1:36
    f_weight_i = f_weight(i);
    for j = 1:n_y_quad
        y = y_quad(j,1);
        par1 = y_quad(j,2);
        par2 = y_quad(j,3);
        f_effective_1 = f_quad(i,1);
        f_effective_2 = f_quad(i,2);
        if par1 == 0
            f_effective_1 = f_effective_1 + parent_params11(1);
            f_effective_2 = f_effective_2 + parent_params12(1);
        elseif par1 == 1
            f_effective_1 = f_effective_1 + parent_params11(2);
            f_effective_2 = f_effective_2 + parent_params12(2);
        else
            f_effective_1 = f_effective_1 + parent_params11(3);
            f_effective_2 = f_effective_2 + parent_params12(3);
        end
        if par2 == 0
            f_effective_1 = f_effective_1 + parent_params21(1);
            f_effective_2 = f_effective_2 + parent_params22(1);
        elseif par2 == 1
            f_effective_1 = f_effective_1 + parent_params21(2);
            f_effective_2 = f_effective_2 + parent_params22(2);
        else
            f_effective_1 = f_effective_1 + parent_params21(3);
            f_effective_2 = f_effective_2 + parent_params22(3);
        end
        if y == 0
            if deriv_index == 1
                d = d + f_weight_i * y_weight(j) * (1 - exp(f_effective_1)/(exp(f_effective_1) + exp(f_effective_2) + 1));
            else
                d = d - f_weight_i * y_weight(j) * exp(f_effective_2)/(exp(f_effective_1) + exp(f_effective_2) + 1);
            end
        elseif y == 1
            if deriv_index == 1
                d = d - f_weight_i * y_weight(j) * exp(f_effective_1)/(exp(f_effective_1) + exp(f_effective_2) + 1);
            else
                d = d + f_weight_i * y_weight(j) * (1 - exp(f_effective_2)/(exp(f_effective_1) + exp(f_effective_2) + 1));
            end
        else
            if deriv_index == 1
                d = d - f_weight_i * y_weight(j) * exp(f_effective_1)/(exp(f_effective_1) + exp(f_effective_2) + 1);
            else
                d = d - f_weight_i * y_weight(j) * exp(f_effective_2)/(exp(f_effective_1) + exp(f_effective_2) + 1);
            end
        end
    end
end

end

function d = ternary_likelihood_var_deriv(deriv_index,f_mean1,f_mean2,f_var1,f_var2,y_quad,y_weight,parent_params11,parent_params21,parent_params12,parent_params22)
hermite_quad = [-2.350604973674492222834; -1.335849074013696949715; -0.4360774119276165086792; 0.4360774119276165086792; 1.335849074013696949715; 2.350604973674492222834];
hermite_weight = [0.0025557844020562465; 0.08861574604191454; 0.40882846955602925; 0.40882846955602925; 0.08861574604191454; 0.0025557844020562465];
f_quad = [];
f_weight = [];
hermite_combined = [];
n_y_quad = length(y_weight);
for i = 1:6
    for j = 1:6
        f_quad = [f_quad; f_mean1 + sqrt(2*f_var1)*hermite_quad(i), f_mean2 + sqrt(2*f_var2)*hermite_quad(j)];
        f_weight = [f_weight; hermite_weight(i)*hermite_weight(j)];
        hermite_combined = [hermite_combined; hermite_quad(i), hermite_quad(j)];
    end
end

d = 0;
for i = 1:36
    f_weight_i = f_weight(i);
    for j = 1:n_y_quad
        y = y_quad(j,1);
        par1 = y_quad(j,2);
        par2 = y_quad(j,3);
        f_effective_1 = f_quad(i,1);
        f_effective_2 = f_quad(i,2);
        if par1 == 0
            f_effective_1 = f_effective_1 + parent_params11(1);
            f_effective_2 = f_effective_2 + parent_params12(1);
        elseif par1 == 1
            f_effective_1 = f_effective_1 + parent_params11(2);
            f_effective_2 = f_effective_2 + parent_params12(2);
        else
            f_effective_1 = f_effective_1 + parent_params11(3);
            f_effective_2 = f_effective_2 + parent_params12(3);
        end
        if par2 == 0
            f_effective_1 = f_effective_1 + parent_params21(1);
            f_effective_2 = f_effective_2 + parent_params22(1);
        elseif par2 == 1
            f_effective_1 = f_effective_1 + parent_params21(2);
            f_effective_2 = f_effective_2 + parent_params22(2);
        else
            f_effective_1 = f_effective_1 + parent_params21(3);
            f_effective_2 = f_effective_2 + parent_params22(3);
        end
        if y == 0
            if deriv_index == 1
                d = d + hermite_combined(i,1)/sqrt(2*f_var1) * f_weight_i * y_weight(j) * (1 - exp(f_effective_1)/(exp(f_effective_1) + exp(f_effective_2) + 1));
            else
                d = d - hermite_combined(i,2)/sqrt(2*f_var2) * f_weight_i * y_weight(j) * exp(f_effective_2)/(exp(f_effective_1) + exp(f_effective_2) + 1);
            end
        elseif y == 1
            if deriv_index == 1
                d = d - hermite_combined(i,1)/sqrt(2*f_var1) * f_weight_i * y_weight(j) * exp(f_effective_1)/(exp(f_effective_1) + exp(f_effective_2) + 1);
            else
                d = d + hermite_combined(i,2)/sqrt(2*f_var2) * f_weight_i * y_weight(j) * (1 - exp(f_effective_2)/(exp(f_effective_1) + exp(f_effective_2) + 1));
            end
        else
            if deriv_index == 1
                d = d - hermite_combined(i,1)/sqrt(2*f_var1) * f_weight_i * y_weight(j) * exp(f_effective_1)/(exp(f_effective_1) + exp(f_effective_2) + 1);
            else
                d = d - hermite_combined(i,2)/sqrt(2*f_var2) * f_weight_i * y_weight(j) * exp(f_effective_2)/(exp(f_effective_1) + exp(f_effective_2) + 1);
            end
        end
    end
end

end

function d = ternary_likelihood_parent_deriv(gp_index,parent_index,param_index,f_mean1,f_mean2,f_var1,f_var2,y_quad,y_weight,parent_params11,parent_params21,parent_params12,parent_params22)
hermite_quad = [-2.350604973674492222834; -1.335849074013696949715; -0.4360774119276165086792; 0.4360774119276165086792; 1.335849074013696949715; 2.350604973674492222834];
hermite_weight = [0.0025557844020562465; 0.08861574604191454; 0.40882846955602925; 0.40882846955602925; 0.08861574604191454; 0.0025557844020562465];
f_quad = [];
f_weight = [];
hermite_combined = [];
n_y_quad = length(y_weight);
for i = 1:6
    for j = 1:6
        f_quad = [f_quad; f_mean1 + sqrt(2*f_var1)*hermite_quad(i), f_mean2 + sqrt(2*f_var2)*hermite_quad(j)];
        f_weight = [f_weight; hermite_weight(i)*hermite_weight(j)];
        hermite_combined = [hermite_combined; hermite_quad(i), hermite_quad(j)];
    end
end

d = 0;
for i = 1:36
    f_weight_i = f_weight(i);
    for j = 1:n_y_quad
        y = y_quad(j,1);
        par1 = y_quad(j,2);
        par2 = y_quad(j,3);
        f_effective_1 = f_quad(i,1);
        f_effective_2 = f_quad(i,2);
        dfdp = 0;
        if par1 == 0
            f_effective_1 = f_effective_1 + parent_params11(1);
            f_effective_2 = f_effective_2 + parent_params12(1);
            if parent_index == 1 && param_index == 1
                dfdp = 1;
            end
        elseif par1 == 1
            f_effective_1 = f_effective_1 + parent_params11(2);
            f_effective_2 = f_effective_2 + parent_params12(2);
            if parent_index == 1 && param_index == 2
                dfdp = 1;
            end
        else
            f_effective_1 = f_effective_1 + parent_params11(3);
            f_effective_2 = f_effective_2 + parent_params12(3);
            if parent_index == 1 && param_index == 3
                dfdp = 1;
            end
        end
        if par2 == 0
            f_effective_1 = f_effective_1 + parent_params21(1);
            f_effective_2 = f_effective_2 + parent_params22(1);
            if parent_index == 2 && param_index == 1
                dfdp = 1;
            end
        elseif par2 == 1
            f_effective_1 = f_effective_1 + parent_params21(2);
            f_effective_2 = f_effective_2 + parent_params22(2);
            if parent_index == 2 && param_index == 2
                dfdp = 1;
            end
        else
            f_effective_1 = f_effective_1 + parent_params21(3);
            f_effective_2 = f_effective_2 + parent_params22(3);
            if parent_index == 2 && param_index == 3
                dfdp = 1;
            end
        end
        if y == 0
            if gp_index == 1
                d = d + dfdp * f_weight_i * y_weight(j) * (1 - exp(f_effective_1)/(exp(f_effective_1) + exp(f_effective_2) + 1));
            else
                d = d - dfdp * f_weight_i * y_weight(j) * exp(f_effective_2)/(exp(f_effective_1) + exp(f_effective_2) + 1);
            end
        elseif y == 1
            if gp_index == 1
                d = d - dfdp * f_weight_i * y_weight(j) * exp(f_effective_1)/(exp(f_effective_1) + exp(f_effective_2) + 1);
            else
                d = d + dfdp * f_weight_i * y_weight(j) * (1 - exp(f_effective_2)/(exp(f_effective_1) + exp(f_effective_2) + 1));
            end
        else
            if gp_index == 1
                d = d - dfdp * f_weight_i * y_weight(j) * exp(f_effective_1)/(exp(f_effective_1) + exp(f_effective_2) + 1);
            else
                d = d - dfdp * f_weight_i * y_weight(j) * exp(f_effective_2)/(exp(f_effective_1) + exp(f_effective_2) + 1);
            end
        end
    end
end

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