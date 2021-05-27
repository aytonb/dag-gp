import GPy
import numpy as np


# Before running this test, download exchange.dat into this folder from
# http://fx.sauder.ubc.ca/.

# Run this test from the command line as
# `python3 slfm_experiment.py`


def make_X(train,j):
    X = np.zeros(251)
    Y = np.zeros(251)
    ind = 0
    for i in range(0,251):
        if not np.isnan(train[i,j]):
            X[ind] = train[i,0]
            Y[ind] = train[i,j]
            ind += 1
    return (X[0:ind][:,None],Y[0:ind][:,None])


if __name__ == "__main__":
    with open('exchange.dat') as f:
        for row in range(0,2):
            f.readline()

        i = -1
        locs = []
        dat = np.zeros((251,14))
        for i in range(0,251):
            s = f.readline()
            vals = s.split(',')
            #vals = list(map(float,s.split(',')))
            dat[i,0] = int(vals[0]) - 2454102
            for j in range(1,14):
                if vals[j+2] == "":
                    dat[i,j] = None
                else:
                    dat[i,j] = float(vals[j+2])
                    

    X = dat[:,0]
    
    train = np.copy(dat)
    test = np.copy(dat)

    for i in range(0,251):
        if i >= 49 and i <= 99:
            train[i,4] = None
        if i >= 99 and i <= 149:
            train[i,10] = None
        if i >= 149 and i <= 199:
            train[i,9] = None


    K1 = GPy.kern.RatQuad(1)
    K2 = GPy.kern.RatQuad(1)
    #K3 = GPy.kern.RatQuad(1)
    #K4 = GPy.kern.RatQuad(1)
    #K5 = GPy.kern.RatQuad(1)
    #K6 = GPy.kern.RatQuad(1)
    LCM = GPy.util.multioutput.LCM(input_dim=1,num_outputs=8,kernels_list=[K1,K2])
                
    XAu,YAu = make_X(train,1)
    XAg,YAg = make_X(train,2)
    XPt,YPt = make_X(train,3)
    XCAD,YCAD = make_X(train,4)
    XEUR,YEUR = make_X(train,5)
    XJPY,YJPY = make_X(train,6)
    XGBP,YGBP = make_X(train,7)
    XCHF,YCHF = make_X(train,8)
    XAUD,YAUD = make_X(train,9)
    XHKD,YHKD = make_X(train,10)
    XNZD,YNZD = make_X(train,11)

    # Normalize
    meanAu = np.mean(YAu)
    meanAg = np.mean(YAg)
    meanPt = np.mean(YPt)
    meanCAD = np.mean(YCAD)
    meanEUR = np.mean(YEUR)
    meanJPY = np.mean(YJPY)
    meanGBP = np.mean(YGBP)
    meanCHF = np.mean(YCHF)
    meanAUD = np.mean(YAUD)
    meanHKD = np.mean(YHKD)
    meanNZD = np.mean(YNZD)
    stdAu = np.std(YAu)
    stdAg = np.std(YAg)
    stdPt = np.std(YPt)
    stdCAD = np.std(YCAD)
    stdEUR = np.std(YEUR)
    stdJPY = np.std(YJPY)
    stdGBP = np.std(YGBP)
    stdCHF = np.std(YCHF)
    stdAUD = np.std(YAUD)
    stdHKD = np.std(YHKD)
    stdNZD = np.std(YNZD)

    YAu = (YAu - meanAu)/stdAu
    YAg = (YAg - meanAg)/stdAg
    YPt = (YPt - meanPt)/stdPt
    YCAD = (YCAD - meanCAD)/stdCAD
    YEUR = (YEUR - meanEUR)/stdEUR
    YJPY = (YJPY - meanJPY)/stdJPY
    YGBP = (YGBP - meanGBP)/stdGBP
    YCHF = (YCHF - meanCHF)/stdCHF
    YAUD = (YAUD - meanAUD)/stdAUD
    YHKD = (YHKD - meanHKD)/stdHKD
    YNZD = (YNZD - meanNZD)/stdNZD

    model = GPy.models.GPCoregionalizedRegression(X_list=[XAu,XAg,XCAD,XEUR,XCHF,XAUD,XHKD,XNZD],Y_list=[YAu,YAg,YCAD,YEUR,YCHF,YAUD,YHKD,YNZD],kernel=LCM)

    model.optimize('lbfgs',messages=True)

    pred_XCAD = test[49:100,0][:,None]
    pred_XCAD = np.hstack([pred_XCAD,2*np.ones_like(pred_XCAD)])
    YCAD_meta = {'output_index':pred_XCAD[:,1].astype(int)}
    pred_YCAD = model.predict(pred_XCAD,Y_metadata=YCAD_meta)

    pred_XHKD = test[99:150,0][:,None]
    pred_XHKD = np.hstack([pred_XHKD,6*np.ones_like(pred_XHKD)])
    YHKD_meta = {'output_index':pred_XHKD[:,1].astype(int)}
    pred_YHKD = model.predict(pred_XHKD,Y_metadata=YHKD_meta)

    pred_XAUD = test[149:200,0][:,None]
    pred_XAUD = np.hstack([pred_XAUD,5*np.ones_like(pred_XAUD)])
    YAUD_meta = {'output_index':pred_XAUD[:,1].astype(int)}
    pred_YAUD = model.predict(pred_XAUD,Y_metadata=YAUD_meta)

    cseCAD = 0
    cse_normCAD = 0
    truthCAD = test[49:100,4]
    mean_truthCAD = np.mean(truthCAD)
    cnllCAD = 0
    for i in range(51):
        pred_mean = meanCAD + pred_YCAD[0][i,0] * stdCAD
        pred_var = pred_YCAD[1][i,0] * stdCAD**2
        cseCAD += (pred_mean - truthCAD[i])**2
        cse_normCAD += (mean_truthCAD - truthCAD[i])**2
        cnllCAD += 0.5 * ((pred_mean - truthCAD[i])**2 / pred_var + np.log(2*np.pi*pred_var))

    cseHKD = 0
    cse_normHKD = 0
    truthHKD = test[99:150,10]
    mean_truthHKD = np.mean(truthHKD)
    cnllHKD = 0
    for i in range(51):
        pred_mean = meanHKD + pred_YHKD[0][i,0] * stdHKD
        pred_var = pred_YHKD[1][i,0] * stdHKD**2
        cseHKD += (pred_mean - truthHKD[i])**2
        cse_normHKD += (mean_truthHKD - truthHKD[i])**2
        cnllHKD += 0.5 * ((pred_mean - truthHKD[i])**2 / pred_var + np.log(2*np.pi*pred_var))

    cseAUD = 0
    cse_normAUD = 0
    truthAUD = test[149:200,9]
    mean_truthAUD = np.mean(truthAUD)
    cnllAUD = 0
    for i in range(51):
        pred_mean = meanAUD + pred_YAUD[0][i,0] * stdAUD
        pred_var = pred_YAUD[1][i,0] * stdAUD**2
        cseAUD += (pred_mean - truthAUD[i])**2
        cse_normAUD += (mean_truthAUD - truthAUD[i])**2
        cnllAUD += 0.5 * ((pred_mean - truthAUD[i])**2 / pred_var + np.log(2*np.pi*pred_var))


    print("smseCAD = {}".format(cseCAD/cse_normCAD))
    print("smseHKD = {}".format(cseHKD/cse_normHKD))
    print("smseAUD = {}".format(cseAUD/cse_normAUD))
    print("average = {}".format((cseCAD/cse_normCAD + cseHKD/cse_normHKD + cseAUD/cse_normAUD)/3))
    print("mnllCAD = {}".format(cnllCAD/51))
    print("mnllHKD = {}".format(cnllHKD/51))
    print("mnllAUD = {}".format(cnllAUD/51))
    print("average = {}".format((cnllCAD/51 + cnllHKD/51 + cnllAUD/51)/3))

    # xx = np.arange(1,365,0.2)[:,None]
    
    # xx_CAD = np.hstack([xx,2*np.ones_like(xx)])
    # yy_CAD_meta = {'output_index':xx_CAD[:,1].astype(int)}
    # yy_CAD = model.predict(xx_CAD,Y_metadata=yy_CAD_meta)

    # xx_HKD = np.hstack([xx,6*np.ones_like(xx)])
    # yy_HKD_meta = {'output_index':xx_HKD[:,1].astype(int)}
    # yy_HKD = model.predict(xx_HKD,Y_metadata=yy_HKD_meta)

    # xx_AUD = np.hstack([xx,5*np.ones_like(xx)])
    # yy_AUD_meta = {'output_index':xx_AUD[:,1].astype(int)}
    # yy_AUD = model.predict(xx_AUD,Y_metadata=yy_AUD_meta)

    # out = np.concatenate((xx,meanCAD+yy_CAD[0]*stdCAD,yy_CAD[1]*stdCAD**2,meanHKD+yy_HKD[0]*stdHKD,yy_HKD[1]*stdHKD**2,meanAUD+yy_AUD[0]*stdAUD,yy_AUD[1]*stdAUD**2),axis=1)
    # np.savetxt('slfm_full_predict.txt',out)
    

# Results:

# Trial 1:
# smseCAD = 1.3953861755836388
# smseHKD = 0.8720444713490856
# smseAUD = 0.03204784684657924
# average = 0.7664928312597677
# mnllCAD = -1.8630556022938165
# mnllHKD = -7.745244504965398
# mnllAUD = -3.6633357132068776
# average = -4.42387860682203

# Trial 2:
# smseCAD = 1.4447490858601957
# smseHKD = 0.9955489025748888
# smseAUD = 0.029306098899458788
# average = 0.8232013624448479
# mnllCAD = -1.5750371095765372
# mnllHKD = -7.7897792839856645
# mnllAUD = -3.7707242084354027
# average = -4.378513533999201

# Trial 3:
# smseCAD = 1.3947500677442988
# smseHKD = 0.8716499473079984
# smseAUD = 0.0320379732392193
# average = 0.7661459960971722
# mnllCAD = -1.8642145569785005
# mnllHKD = -7.745429602338385
# mnllAUD = -3.6631436027756767
# average = -4.424262587364187

# Trial 4:
# smseCAD = 2.1359140810102
# smseHKD = 1.6661383099538805
# smseAUD = 0.25670422076363647
# average = 1.3529188705759057
# mnllCAD = -1.3355275325424731
# mnllHKD = -7.570369932358855
# mnllAUD = -2.6127041212815465
# average = -3.8395338620609585

# Trial 5:
# smseCAD = 2.1410109272402775
# smseHKD = 1.6669453411404898
# smseAUD = 0.25750301464996805
# average = 1.3551530943435786
# mnllCAD = -1.3217631599291466
# mnllHKD = -7.5705288102987005
# mnllAUD = -2.607685876555922
# average = -3.8333259489279228

# Trial 6:
# smseCAD = 1.3948523406025513
# smseHKD = 0.8715303406142131
# smseAUD = 0.032042900807712604
# average = 0.7661418606748257
# mnllCAD = -1.863905770842742
# mnllHKD = -7.745341035912679
# mnllAUD = -3.663306333466374
# average = -4.424184380073932

# Trial 7:
# smseCAD = 2.1360526024841766
# smseHKD = 1.6579608446241634
# smseAUD = 0.25730961111611406
# average = 1.3504410194081515
# mnllCAD = -1.3309436235795369
# mnllHKD = -7.571637464530047
# mnllAUD = -2.6095411443242367
# average = -3.8373740774779406

# Trial 8:
# smseCAD = 1.4445803857775867
# smseHKD = 0.9959694442600675
# smseAUD = 0.029372511089696466
# average = 0.8233074470424503
# mnllCAD = -1.5755099822707301
# mnllHKD = -7.789690229654025
# mnllAUD = -3.7700418729692
# average = -4.378414028297985

# Trial 9:
# smseCAD = 1.3951501314380799
# smseHKD = 0.8715639098964566
# smseAUD = 0.03204976690926925
# average = 0.7662546027479352
# mnllCAD = -1.8637191064583345
# mnllHKD = -7.7451732990220465
# mnllAUD = -3.663018516388163
# average = -4.4239703072895145

# Trial 10:
# smseCAD = 1.4447835522316335
# smseHKD = 0.9967247421121028
# smseAUD = 0.029294460490154422
# average = 0.8236009182779637
# mnllCAD = -1.5747256682366442
# mnllHKD = -7.789483545726469
# mnllAUD = -3.7709675598151677
# average = -4.378392257926094


# Averages:
# smseCAD = 1.632722934997264 +/- 0.10473465733096972
# smseHKD = 1.1466076253833346 +/- 0.10827854228029715
# smseAUD = 0.09876684048118085 +/- 0.03279502953524842
# average = 0.9593658002872598 +/- 0.08179960033247602
# mnllCAD = -1.6168402112708464 +/- 0.07049681804714869
# mnllHKD = -7.706267770879227 +/- 0.02863042908483155
# mnllAUD = -3.379446894921857 +/- 0.15991500704227318




