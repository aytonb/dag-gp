clear all;

kern1_param1 = 1.25;
kern1_param2 = 0.7;
kern21_param1 = 0.75;
kern21_param2 = 1.2;
kern22_param1 = 0.3;
kern22_param2 = 1.1;
kern31_param1 = 0.8;
kern31_param2 = 0.9;
kern32_param1 = 0.4;
kern32_param2 = 1.6;

noise1_param = -1.2;

parent11_param1 = 1;
parent21_param1 = -1;
parent21_param2 = -2;
parent21_param3 = -3;
parent12_param1 = 0.5;
parent22_param1 = 3;
parent22_param2 = 0;
parent22_param3 = 2.3;

q_mean1_param1 = 0.3;
q_mean1_param2 = 0.5;
q_mean21_param1 = 1.3;
q_mean21_param2 = 0.6;
q_mean22_param1 = 0.8;
q_mean22_param2 = 0.5;
q_mean31_param1 = 0.9;
q_mean31_param2 = 0.5;
q_mean32_param1 = 2.3;
q_mean32_param2 = 1.3;

q_chol1_param1 = 0.7;
q_chol1_param2 = 0.2;
q_chol1_param3 = 0.8;
q_chol21_param1 = 1;
q_chol21_param2 = 0.2;
q_chol21_param3 = 1.2;
q_chol22_param1 = 0.6;
q_chol22_param2 = 1.4;
q_chol22_param3 = 0.3;
q_chol31_param1 = 0.4;
q_chol31_param2 = 0.1;
q_chol31_param3 = 0.7;
q_chol32_param1 = 0.8;
q_chol32_param2 = 0.3;
q_chol32_param3 = 1.1;

parent_params11 = [parent11_param1];
parent_params21 = [parent21_param1;parent21_param2;parent21_param3];
parent_params12 = [parent12_param1];
parent_params22 = [parent22_param1;parent22_param2;parent22_param3];


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

%% Prediction at (0, 2, 5)

Kff1 = kern1(ff_sq_dist,kern1_param1,kern1_param2);
Kff21 = kern1(ff_sq_dist,kern21_param1,kern21_param2);
Kff22 = kern1(ff_sq_dist,kern22_param1,kern22_param2);
Kff31 = kern1(ff_sq_dist,kern31_param1,kern31_param2);
Kff32 = kern1(ff_sq_dist,kern32_param1,kern32_param2);
% Kuu has added jitter
Kuu1 = kern1(uu_sq_dist,kern1_param1,kern1_param2) + 1e-5 * eye(2);
Kuu21 = kern1(uu_sq_dist,kern21_param1,kern21_param2) + 1e-5 * eye(2);
Kuu22 = kern1(uu_sq_dist,kern22_param1,kern22_param2) + 1e-5 * eye(2);
Kuu31 = kern1(uu_sq_dist,kern31_param1,kern31_param2) + 1e-5 * eye(2);
Kuu32 = kern1(uu_sq_dist,kern32_param1,kern32_param2) + 1e-5 * eye(2);

Kfu1 = kern1(fu_sq_dist,kern1_param1,kern1_param2);
Kfu21 = kern1(fu_sq_dist,kern21_param1,kern21_param2);
Kfu22 = kern1(fu_sq_dist,kern22_param1,kern22_param2);
Kfu31 = kern1(fu_sq_dist,kern31_param1,kern31_param2);
Kfu32 = kern1(fu_sq_dist,kern32_param1,kern32_param2);

q_chol1 = [q_chol1_param1,0;q_chol1_param2,q_chol1_param3];
q_cov1 = q_chol1 * q_chol1';
q_chol21 = [q_chol21_param1,0;q_chol21_param2,q_chol21_param3];
q_cov21 = q_chol21 * q_chol21';
q_chol22 = [q_chol22_param1,0;q_chol22_param2,q_chol22_param3];
q_cov22 = q_chol22 * q_chol22';
q_chol31 = [q_chol31_param1,0;q_chol31_param2,q_chol31_param3];
q_cov31 = q_chol31 * q_chol31';
q_chol32 = [q_chol32_param1,0;q_chol32_param2,q_chol32_param3];
q_cov32 = q_chol32 * q_chol32';

q_mean1 = [q_mean1_param1; q_mean1_param2];
qf_mean1 = Kfu1 * (Kuu1 \ q_mean1);
q_mean21 = [q_mean21_param1; q_mean21_param2];
qf_mean21 = Kfu21 * (Kuu21 \ q_mean21);
q_mean22 = [q_mean22_param1; q_mean22_param2];
qf_mean22 = Kfu22 * (Kuu22 \ q_mean22);
q_mean31 = [q_mean31_param1; q_mean31_param2];
qf_mean31 = Kfu31 * (Kuu31 \ q_mean31);
q_mean32 = [q_mean32_param1; q_mean32_param2];
qf_mean32 = Kfu32 * (Kuu32 \ q_mean32);

Kfu_post1 = (Kuu1' \ Kfu1')';
middle1 = q_cov1 - Kuu1; 
KfuM1 = Kfu_post1 * middle1;
Kfu_post21 = (Kuu21' \ Kfu21')';
middle21 = q_cov21 - Kuu21; 
KfuM21 = Kfu_post21 * middle21;
Kfu_post22 = (Kuu22' \ Kfu22')';
middle22 = q_cov22 - Kuu22; 
KfuM22 = Kfu_post22 * middle22;
Kfu_post31 = (Kuu31' \ Kfu31')';
middle31 = q_cov31 - Kuu31; 
KfuM31 = Kfu_post31 * middle31;
Kfu_post32 = (Kuu32' \ Kfu32')';
middle32 = q_cov32 - Kuu32; 
KfuM32 = Kfu_post32 * middle32;
 
Kff_post1 = KfuM1 * Kfu_post1';
qf_cov1 = Kff1 + diag(Kff_post1);
Kff_post21 = KfuM21 * Kfu_post21';
qf_cov21 = Kff21 + diag(Kff_post21);
Kff_post22 = KfuM22 * Kfu_post22';
qf_cov22 = Kff22 + diag(Kff_post22);
Kff_post31 = KfuM31 * Kfu_post31';
qf_cov31 = Kff31 + diag(Kff_post31);
Kff_post32 = KfuM32 * Kfu_post32';
qf_cov32 = Kff32 + diag(Kff_post32);

% Observations are 0: (0,1.5),(2,4), 1: (2,1),(5,0), 2: (5,2)
pred_probs_0 = predict_output_0(true,1.5,qf_mean1(1),qf_cov1(1),noise1_param);
pred_probs_0 = predict_output_2(pred_probs_0,false,false,qf_mean31(1),qf_mean32(1),qf_cov31(1),qf_cov32(1));
pred_probs_0 = predict_output_1(pred_probs_0,false,false,qf_mean21(1),qf_mean22(1),qf_cov21(1),qf_cov22(1),parent_params11,parent_params21,parent_params12,parent_params22);

pred_probs_2 = predict_output_0(true,4,qf_mean1(2),qf_cov1(2),noise1_param);
pred_probs_2 = predict_output_2(pred_probs_2,false,false,qf_mean31(2),qf_mean32(2),qf_cov31(2),qf_cov32(2));
pred_probs_2 = predict_output_1(pred_probs_2,true,1,qf_mean21(2),qf_mean22(2),qf_cov21(2),qf_cov22(2),parent_params11,parent_params21,parent_params12,parent_params22);

pred_probs_5 = predict_output_0(false,false,qf_mean1(3),qf_cov1(3),noise1_param);
pred_probs_5 = predict_output_2(pred_probs_5,true,2,qf_mean31(3),qf_mean32(3),qf_cov31(3),qf_cov32(3));
pred_probs_5 = predict_output_1(pred_probs_5,true,0,qf_mean21(3),qf_mean22(3),qf_cov21(3),qf_cov22(3),parent_params11,parent_params21,parent_params12,parent_params22);

function out = kern1(in,param1,param2)

out = exp(param1).*exp(-in./exp(param2));

end

function new_pred_probs = predict_output_0(y0_known,y0,f_mean,f_var,noise_param)
eps_var = exp(noise_param);
hermite_quad = [-2.350604973674492222834; -1.335849074013696949715; -0.4360774119276165086792; 0.4360774119276165086792; 1.335849074013696949715; 2.350604973674492222834];
hermite_weight = [0.0025557844020562465; 0.08861574604191454; 0.40882846955602925; 0.40882846955602925; 0.08861574604191454; 0.0025557844020562465];
if y0_known
    new_pred_probs = [y0, 1];
else
    effective_var = f_var + eps_var;
    new_pred_probs = [];
    for i = 1:6
        new_pred_probs = [new_pred_probs; f_mean + sqrt(2*effective_var)*hermite_quad(i), hermite_weight(i)];
    end
end
end

function new_pred_probs = predict_output_2(pred_probs,y2_known,y2,f_mean1,f_mean2,f_var1,f_var2)
hermite_quad = [-2.350604973674492222834; -1.335849074013696949715; -0.4360774119276165086792; 0.4360774119276165086792; 1.335849074013696949715; 2.350604973674492222834];
hermite_weight = [0.0025557844020562465; 0.08861574604191454; 0.40882846955602925; 0.40882846955602925; 0.08861574604191454; 0.0025557844020562465];
[rows,~] = size(pred_probs);

if y2_known
    new_pred_probs = [pred_probs(:,1), y2*ones(rows,1), pred_probs(:,2)];
else
    f_quad = [];
    f_weight = [];
    for i = 1:6
        for j = 1:6         
            f_quad = [f_quad; f_mean1 + sqrt(2*f_var1)*hermite_quad(i), f_mean2 + sqrt(2*f_var2)*hermite_quad(j)];
            f_weight = [f_weight; hermite_weight(i)*hermite_weight(j)];
        end
    end
    
    new_pred_probs = [];
    for r = 1:rows
        base_weight = pred_probs(r,2);
        new_rows = [pred_probs(r,1),0,0;
                    pred_probs(r,1),1,0;
                    pred_probs(r,1),2,0];
        for i = 1:36
            new_rows(1,3) = new_rows(1,3) + base_weight * f_weight(i) * exp(f_quad(i,1))/(exp(f_quad(i,1))+exp(f_quad(i,2))+1);
            new_rows(2,3) = new_rows(2,3) + base_weight * f_weight(i) * exp(f_quad(i,2))/(exp(f_quad(i,1))+exp(f_quad(i,2))+1);
            new_rows(3,3) = new_rows(3,3) + base_weight * f_weight(i) * 1/(exp(f_quad(i,1))+exp(f_quad(i,2))+1);
        end
        new_pred_probs = [new_pred_probs; new_rows];
    end
end
end

function new_pred_probs = predict_output_1(pred_probs,y1_known,y1,f_mean1,f_mean2,f_var1,f_var2,parent_params11,parent_params21,parent_params12,parent_params22)
hermite_quad = [-2.350604973674492222834; -1.335849074013696949715; -0.4360774119276165086792; 0.4360774119276165086792; 1.335849074013696949715; 2.350604973674492222834];
hermite_weight = [0.0025557844020562465; 0.08861574604191454; 0.40882846955602925; 0.40882846955602925; 0.08861574604191454; 0.0025557844020562465];
[rows,~] = size(pred_probs);
hermite_combined = [];

if y1_known
    new_pred_probs = [pred_probs(:,1), y1*ones(rows,1), pred_probs(:,2:3)];
else
    f_quad = [];
    f_weight = [];
    for i = 1:6
        for j = 1:6
            f_quad = [f_quad; f_mean1 + sqrt(2*f_var1)*hermite_quad(i), f_mean2 + sqrt(2*f_var2)*hermite_quad(j)];
            f_weight = [f_weight; hermite_weight(i)*hermite_weight(j)];
            hermite_combined = [hermite_combined; hermite_quad(i), hermite_quad(j)];
        end
    end
    
    new_pred_probs = [];
    for r = 1:rows
        base_weight = pred_probs(r,3);
        new_rows = [pred_probs(r,1),0,pred_probs(r,2),0;
                    pred_probs(r,1),1,pred_probs(r,2),0;
                    pred_probs(r,1),2,pred_probs(r,2),0];
        for i = 1:36
            f_effective_1 = f_quad(i,1) + pred_probs(r,1) * parent_params11(1);
            f_effective_2 = f_quad(i,2) + pred_probs(r,1) * parent_params12(1);
           
            if pred_probs(r,2) == 0
                f_effective_1 = f_effective_1 + parent_params21(1);
                f_effective_2 = f_effective_2 + parent_params22(1);
            elseif pred_probs(r,2) == 1
                f_effective_1 = f_effective_1 + parent_params21(2);
                f_effective_2 = f_effective_2 + parent_params22(2);
            else
                f_effective_1 = f_effective_1 + parent_params21(3);
                f_effective_2 = f_effective_2 + parent_params22(3);
            end
            fprintf('f_mean1 = %f, f_var1 = %f\n',[f_mean1,f_var1]);
            fprintf('f_quad = [%f %f]\n',f_quad(i,:));
            fprintf('parent_vals = [%f %f]\n',pred_probs(r,1:2));
            fprintf('parent_bases = [%f %f]\n',[f_effective_1-f_quad(i,1),f_effective_2-f_quad(i,2)]);
            fprintf('f_effective = [%f %f]\n',[f_effective_1,f_effective_2]);
            fprintf('weight = %e\n',f_weight(i));
            fprintf('point = [%f %f]\n',hermite_combined(i,:));
            fprintf('f-list = [%f %f]\n',[exp(f_effective_1),exp(f_effective_2)]);
            new_rows(1,4) = new_rows(1,4) + base_weight * f_weight(i) * exp(f_effective_1)/(exp(f_effective_1)+exp(f_effective_2)+1);
            new_rows(2,4) = new_rows(2,4) + base_weight * f_weight(i) * exp(f_effective_2)/(exp(f_effective_1)+exp(f_effective_2)+1);
            new_rows(3,4) = new_rows(3,4) + base_weight * f_weight(i) * 1/(exp(f_effective_1)+exp(f_effective_2)+1);
        end
        new_pred_probs = [new_pred_probs; new_rows];
    end
end
end





