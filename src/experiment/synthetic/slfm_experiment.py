import GPy
import numpy as np
from matplotlib import pyplot


# Run this test from the command line as
# `python3 slfm_experiment.py`


def make_dat():
    x = np.linspace(0,10,41)
    
    y1 = np.exp(-(x-2)**2)
    y2 = 5*np.exp(-0.4*(x-2.5)**2) + 3*np.exp(-4*(x-7)**2) + 0.01*x**3 + 1.1*np.exp(-x**2)
    y3 = 2*y1 + 0.8*np.log(x+0.25)

    return np.hstack((x[:,None],y1[:,None],y2[:,None],y3[:,None]))


def make_X(train,j):
    X = np.zeros(41)
    Y = np.zeros(41)
    ind = 0
    for i in range(0,41):
        if not np.isnan(train[i,j]):
            X[ind] = train[i,0]
            Y[ind] = train[i,j]
            ind += 1
    return (X[0:ind][:,None],Y[0:ind][:,None])


if __name__ == "__main__":
    train = make_dat()
    test = make_dat()

    for i in range(0,41):
        if i >= 24 and i <= 32:
            train[i,3] = None

    K1 = GPy.kern.RBF(1)
    K2 = GPy.kern.RBF(1)
    K3 = GPy.kern.RBF(1)
    K4 = GPy.kern.RBF(1)

    LCM = GPy.util.multioutput.LCM(input_dim=1,num_outputs=3,kernels_list=[K1,K2])

    X0,Y0 = make_X(train,1)
    X1,Y1 = make_X(train,2)
    X2,Y2 = make_X(train,3)

    # Normalize
    mean0 = np.mean(Y0)
    mean1 = np.mean(Y1)
    mean2 = np.mean(Y2)
    std0 = np.std(Y0)
    std1 = np.std(Y1)
    std2 = np.std(Y2)

    Y0 = (Y0 - mean0)/std0
    Y1 = (Y1 - mean1)/std1
    Y2 = (Y2 - mean2)/std2

    model = GPy.models.GPCoregionalizedRegression(X_list=[X0,X1,X2],Y_list=[Y0,Y1,Y2],kernel=LCM)

    model.optimize('lbfgs',messages=True)

    pred_X2 = test[24:33,0][:,None]
    pred_X2 = np.hstack([pred_X2,2*np.ones_like(pred_X2)])
    Y2_meta = {'output_index':pred_X2[:,1].astype(int)}
    pred_Y2= model.predict(pred_X2,Y_metadata=Y2_meta)

    # Cumulative errors
    cae2 = 0
    cse2 = 0
    truth2 = test[24:33,3]
    for i in range(9):
        cae2 += abs(mean2 + pred_Y2[0][i,0] * std2 - truth2[i])
        cse2 += (mean2 + pred_Y2[0][i,0] * std2 - truth2[i])**2
        print("pred = {}, truth = {}".format(mean2 + pred_Y2[0][i,0] * std2,truth2[i]))
        
    print("cae2 = {}".format(cae2))
    print("cse2 = {}".format(cse2))


    # xx = np.arange(0,10.05,0.05)[:,None]

    # XX0 = np.hstack([xx,0*np.ones_like(xx)])
    # YY0_meta = {'output_index':XX0[:,1].astype(int)}
    # YY0 = model.predict(XX0,Y_metadata=YY0_meta)

    # XX1 = np.hstack([xx,1*np.ones_like(xx)])
    # YY1_meta = {'output_index':XX1[:,1].astype(int)}
    # YY1 = model.predict(XX1,Y_metadata=YY1_meta)

    # XX2 = np.hstack([xx,2*np.ones_like(xx)])
    # YY2_meta = {'output_index':XX2[:,1].astype(int)}
    # YY2 = model.predict(XX2,Y_metadata=YY2_meta)
   
    # out = np.concatenate((xx,mean0+YY0[0]*std0,YY0[1]*std0**2,mean1+YY1[0]*std1,YY1[1]*std1**2,mean2+YY2[0]*std2,YY2[1]*std2**2),axis=1)
    # np.savetxt('slfm_full_predict_2.txt',out)

    # str = ''
    # str += "(list"
    # for i in range(41):
    #     str += " {:e}".format(test[i,1])
    # str += ")"
    # str = str.replace("e+","d")
    # str = str.replace("e","d")
    # print(str)

    # str = ''
    # str += "(list"
    # for i in range(41):
    #     str += " {:e}".format(test[i,2])
    # str += ")"
    # str = str.replace("e+","d")
    # str = str.replace("e","d")
    # print(str)

    # str = ''
    # str += "(list"
    # for i in range(41):
    #     str += " {:e}".format(test[i,3])
    # str += ")"
    # str = str.replace("e+","d")
    # str = str.replace("e","d")
    # print(str)
        

# Results:

# Trial 1:
# pred = 1.49137844823286, truth = 1.4660653960689978
# pred = 1.612470792695413, truth = 1.497441770135757
# pred = 1.805489246601131, truth = 1.527634007118007
# pred = 2.007953937228308, truth = 1.5567281195621288
# pred = 2.1162286281475167, truth = 1.5848011751210425
# pred = 2.074624715897187, truth = 1.6119224164359536
# pred = 1.9297584163284727, truth = 1.63815427469235
# pred = 1.786475008194492, truth = 1.6635532333438774
# pred = 1.7153728224716476, truth = 1.688170560277272
# cae2 = 2.3052810630416416
# cse2 = 0.8920719562547913

# Trial 2:
# pred = 1.499840800569184, truth = 1.4660653960689978
# pred = 1.798039726761157, truth = 1.497441770135757
# pred = 2.7301226551996036, truth = 1.527634007118007
# pred = 4.228497159683908, truth = 1.5567281195621288
# pred = 5.056078549165892, truth = 1.5848011751210425
# pred = 4.279091481706928, truth = 1.6119224164359536
# pred = 2.8347976625229796, truth = 1.63815427469235
# pred = 1.9603500333603483, truth = 1.6635532333438774
# pred = 1.7207515460481215, truth = 1.688170560277272
# cae2 = 11.873098662262736
# cse2 = 29.36049135106161

# Trial 3:
# pred = 1.4441593728214919, truth = 1.4660653960689978
# pred = 1.2962509359741168, truth = 1.497441770135757
# pred = 0.7128425801979846, truth = 1.527634007118007
# pred = -0.26262380799579566, truth = 1.5567281195621288
# pred = -0.7829498017070371, truth = 1.5848011751210425
# pred = -0.20680195716601957, truth = 1.6119224164359536
# pred = 0.8240618255625544, truth = 1.63815427469235
# pred = 1.4627570538945043, truth = 1.6635532333438774
# pred = 1.6663814799850196, truth = 1.688170560277272
# cae2 = 8.080393271188568
# cse2 = 13.63242755213209

# Trial 4:
# pred = 1.4997924613426483, truth = 1.4660653960689978
# pred = 1.797622094765203, truth = 1.497441770135757
# pred = 2.728452208692878, truth = 1.527634007118007
# pred = 4.224776823364594, truth = 1.5567281195621288
# pred = 5.0512599002012655, truth = 1.5848011751210425
# pred = 4.27543852538038, truth = 1.6119224164359536
# pred = 2.8332135413455948, truth = 1.63815427469235
# pred = 1.9599883018660287, truth = 1.6635532333438774
# pred = 1.7207198078838097, truth = 1.688170560277272
# cae2 = 11.856792712087016
# cse2 = 29.27944799079667

# Trial 5:
# pred = 1.4428861510647353, truth = 1.4660653960689978
# pred = 1.2855697463844833, truth = 1.497441770135757
# pred = 0.6715789471948714, truth = 1.527634007118007
# pred = -0.35290494886982926, truth = 1.5567281195621288
# pred = -0.89976729843311, truth = 1.5848011751210425
# pred = -0.2970820627210278, truth = 1.6119224164359536
# pred = 0.7828746556094288, truth = 1.63815427469235
# pred = 1.452198476307983, truth = 1.6635532333438774
# pred = 1.6651849111631232, truth = 1.688170560277272
# cae2 = 8.48393237505473
# cse2 = 15.019036754942443

# Trial 6:
# pred = 1.5048339752952449, truth = 1.4660653960689978
# pred = 1.6770557811089086, truth = 1.497441770135757
# pred = 1.9660276283873785, truth = 1.527634007118007
# pred = 2.2712277396404703, truth = 1.5567281195621288
# pred = 2.4241559930316634, truth = 1.5848011751210425
# pred = 2.336458053470614, truth = 1.6119224164359536
# pred = 2.087924510198634, truth = 1.63815427469235
# pred = 1.8488633174337663, truth = 1.6635532333438774
# pred = 1.7277931678890637, truth = 1.688170560277272
# cae2 = 3.6098692137003585
# cse2 = 2.204134312664584

# Trial 7:
# pred = 1.4992263550885074, truth = 1.4660653960689978
# pred = 1.7928374971508592, truth = 1.497441770135757
# pred = 2.7097163696469746, truth = 1.527634007118007
# pred = 4.183509747656119, truth = 1.5567281195621288
# pred = 4.997814882323098, truth = 1.5848011751210425
# pred = 4.234379524427945, truth = 1.6119224164359536
# pred = 2.8147437855384516, truth = 1.63815427469235
# pred = 1.9553764527029247, truth = 1.6635532333438774
# pred = 1.720205777603593, truth = 1.688170560277272
# cae2 = 11.673339439383087
# cse2 = 28.38215249058219

# Trial 8:
# pred = 1.499532856257811, truth = 1.4660653960689978
# pred = 1.795734927208307, truth = 1.497441770135757
# pred = 2.7216591142896425, truth = 1.527634007118007
# pred = 4.21039815434872, truth = 1.5567281195621288
# pred = 5.0327289112148925, truth = 1.5848011751210425
# pred = 4.260778294463735, truth = 1.6119224164359536
# pred = 2.8260631087232153, truth = 1.63815427469235
# pred = 1.9578724071892797, truth = 1.6635532333438774
# pred = 1.7203900498330205, truth = 1.688170560277272
# cae2 = 11.790686870773238
# cse2 = 28.96119189398474

# Trial 9:
# pred = 1.4424426363633005, truth = 1.4660653960689978
# pred = 1.2816321422979622, truth = 1.497441770135757
# pred = 0.6558152488938325, truth = 1.527634007118007
# pred = -0.3879264674341323, truth = 1.5567281195621288
# pred = -0.9452709911602333, truth = 1.5848011751210425
# pred = -0.3320411951365112, truth = 1.6119224164359536
# pred = 0.7671764763535533, truth = 1.63815427469235
# pred = 1.4482902211960222, truth = 1.6635532333438774
# pred = 1.6647431057820792, truth = 1.688170560277272
# cae2 = 8.639609775599512
# cse2 = 15.574630265092019

# Trial 10:
# pred = 1.499526736995873, truth = 1.4660653960689978
# pred = 1.7957174877464546, truth = 1.497441770135757
# pred = 2.7216255048844227, truth = 1.527634007118007
# pred = 4.210339051356737, truth = 1.5567281195621288
# pred = 5.032681535758574, truth = 1.5848011751210425
# pred = 4.26079876701845, truth = 1.6119224164359536
# pred = 2.8261334950080483, truth = 1.63815427469235
# pred = 1.9579240317959532, truth = 1.6635532333438774
# pred = 1.720405203566087, truth = 1.688170560277272
# cae2 = 11.790680861375213
# cse2 = 28.960767508187537


# Average mae = 1.0011520471607345 +/- 0.11876531329511687
