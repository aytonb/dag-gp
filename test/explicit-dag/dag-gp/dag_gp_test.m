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

% f = 0, 2, 5
ff_sq_dist = [0,4,25;
              4,0,9;
              25,9,0];

% f pred = 6, 7
pred_ff_sq_dist1 = [0];
pred_ff_sq_dist2 = [0];

pred_fu_sq_dist1 = [25,9];
pred_fu_sq_dist2 = [36,16];


Kff1 = kern1(ff_sq_dist,kern1_param1,kern1_param2) + 1e-5 * eye(3) + exp(noise1_param) * eye(3);
Kff2 = kern1(ff_sq_dist,kern2_param1,kern2_param2) + 1e-5 * eye(3) + exp(noise2_param) * eye(3);
Kff3 = kern1(ff_sq_dist,kern3_param1,kern3_param2) + 1e-5 * eye(3) + exp(noise3_param) * eye(3);

Kff = [Kff1, zeros(3,6);
       zeros(3,3), Kff2, zeros(3,3);
       zeros(3,6), Kff3];
   
a = [1*eye(3),zeros(3,6);
     -parent_param1*eye(3),1*eye(3),-parent_param2*eye(3);
     zeros(3,6),1*eye(3)];
a = inv(a);

Kyy = a * Kff * a';

% Add measurements 0: (0,1),(2,2), 1: (2,1.5),(5,4), 2: (5,2.5)
Kyy_pred = Kyy([3,4,7,8],[3,4,7,8]);
Kyy_obs = Kyy([1,2,5,6,9],[1,2,5,6,9]);
Kyy_pred_obs = Kyy([3,4,7,8],[1,2,5,6,9]);

y_obs = [1;2;1.5;4;2.5];

Kyy = Kyy_pred - Kyy_pred_obs * (Kyy_obs \ Kyy_pred_obs');
Kyy_all = zeros(9,9);
Kyy_all(3:4,3:4) = Kyy(1:2,1:2);
Kyy_all(7:8,7:8) = Kyy(3:4,3:4);
Kyy_all(3:4,7:8) = Kyy(1:2,3:4);
Kyy_all(7:8,3:4) = Kyy(3:4,1:2);
Kyy_all

y_pred =  Kyy_pred_obs * (Kyy_obs \ y_obs);
y_pred_all = [1;2;y_pred([1,2]);1.5;4;y_pred([3,4]);2.5]


%% Second test

 % f = 0, 1, 2, 3, 5
ff_sq_dist2 = [0,1,4,9,25;
               1,0,1,4,16;
               4,1,0,1,9;
               9,4,1,0,4;
               25,16,9,4,0];

Kff12 = kern1(ff_sq_dist2,kern1_param1,kern1_param2) + 1e-5 * eye(5) + exp(noise1_param) * eye(5);
Kff22 = kern1(ff_sq_dist2,kern2_param1,kern2_param2) + 1e-5 * eye(5) + exp(noise2_param) * eye(5);
Kff32 = kern1(ff_sq_dist2,kern3_param1,kern3_param2) + 1e-5 * eye(5) + exp(noise3_param) * eye(5);

Kff2 = [Kff12, zeros(5,10);
       zeros(5,5), Kff22, zeros(5,5);
       zeros(5,10), Kff32];
   
a2 = [1*eye(5),zeros(5,10);
     -parent_param1*eye(5),1*eye(5),-parent_param2*eye(5);
     zeros(5,10),1*eye(5)];
a2 = inv(a2);

Kyy2 = a2 * Kff2 * a2';

% Predict locs 1, 3, 5
% Add measurements 0: (0,1),(2,2), 1: (2,1.5),(5,4), 2: (5,2.5)
Kyy_pred2 = Kyy2([2,4,5,7,9,12,14],[2,4,5,7,9,12,14]);
Kyy_obs2 = Kyy2([1,3,8,10,15],[1,3,8,10,15]);
Kyy_pred_obs2 = Kyy2([2,4,5,7,9,12,14],[1,3,8,10,15]);

y_obs2 = [1;2;1.5;4;2.5];

Kyy2 = Kyy_pred2 - Kyy_pred_obs2 * (Kyy_obs2 \ Kyy_pred_obs2');
% Observed only output 1 and 2 at location 5
Kyy2_all = zeros(9,9);
Kyy2_all([1,2,3,4,5,7,8],[1,2,3,4,5,7,8]) = Kyy2;
Kyy2_all

y_pred2 = Kyy_pred_obs2 * (Kyy_obs2 \ y_obs2);
y_pred2_all = [y_pred2([1:5]);4;y_pred2([6:7]);2.5]

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