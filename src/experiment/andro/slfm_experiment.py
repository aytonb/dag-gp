import GPy
import numpy as np


# Before running this test, ensure andro.arff, downloaded from
# http://mulan.sourceforge.net/datasets-mtr.html is in this folder.

# Run this test from the command line using
# `python3 slfm_experiment.py` 


if __name__ == "__main__":
    with open('andro.arff') as f:
        for row in range(0,40):
            f.readline()

        i = -1
        locs = []
        dat = np.zeros((54,6))
        for i in range(0,48):
            s = f.readline()
            vals = list(map(float,s.split(',')))
            dat[i,0] = vals[0]
            dat[i,1] = vals[1]
            dat[i,2] = vals[2]
            dat[i,3] = vals[3]
            dat[i,4] = vals[4]
            dat[i,5] = vals[5]

        s = f.readline()
        vals = list(map(float,s.split(',')))
        for i in range(0,6):
            for v in range(0,6):
                dat[i+48,v] = vals[v + 6*i]

    X = np.linspace(0,53,54)
    

    train = np.copy(dat)
    test = np.copy(dat)
        
    K1 = GPy.kern.RBF(1)
    K2 = GPy.kern.RBF(1)
    K3 = GPy.kern.RBF(1)
    K4 = GPy.kern.RBF(1)
    K5 = GPy.kern.RBF(1)
    K6 = GPy.kern.RBF(1)
    # Best performance achieved with 6 kernels
    LCM = GPy.util.multioutput.LCM(input_dim=1,num_outputs=6,kernels_list=[K1,K2,K3,K4,K5,K6])
                
    XTemp = X[:,None]
    YTemp = train[:,0][:,None]
    XPH = X[:,None]
    YPH = train[:,1][:,None]
    XCond = X[:,None]
    YCond = train[:,2][:,None]
    XSal = np.concatenate((X[0:20],X[30:]),axis=0)[:,None]
    YSal = np.concatenate((train[0:20,3],train[30:,3]),axis=0)[:,None]
    XOxy = np.concatenate((X[0:30],X[40:]),axis=0)[:,None]
    YOxy = np.concatenate((train[0:30:,4],train[40:,4]),axis=0)[:,None]
    XTurb = X[:,None]
    YTurb = train[:,5][:,None]

    # Normalize
    meanTemp = np.mean(YTemp)
    meanPH   = np.mean(YPH)
    meanCond = np.mean(YCond)
    meanSal  = np.mean(YSal)
    meanOxy  = np.mean(YOxy)
    meanTurb = np.mean(YTurb)
    stdTemp  = np.std(YTemp)
    stdPH    = np.std(YPH)
    stdCond  = np.std(YCond)
    stdSal   = np.std(YSal)
    stdOxy   = np.std(YOxy)
    stdTurb  = np.std(YTurb)

    YTemp = (YTemp - meanTemp) /stdTemp
    YPH   = (YPH   - meanPH)   /stdPH
    YCond = (YCond - meanCond) /stdCond
    YSal  = (YSal  - meanSal)  /stdSal
    YOxy  = (YOxy  - meanOxy)  /stdOxy
    YTurb = (YTurb - meanTurb) /stdTurb

    model = GPy.models.GPCoregionalizedRegression(X_list=[XTemp,XPH,XCond,XSal,XOxy,XTurb],Y_list=[YTemp,YPH,YCond,YSal,YOxy,YTurb],kernel=LCM)

    model.optimize('lbfgs',messages=True)

    pred_XSal = np.arange(20,30)[:,None]
    pred_XSal = np.hstack([pred_XSal,3*np.ones_like(pred_XSal)])
    YSal_meta = {'output_index':pred_XSal[:,1].astype(int)}

    pred_YSal = model.predict(pred_XSal,Y_metadata=YSal_meta)
    
    pred_XOxy = np.arange(30,40)[:,None]
    pred_XOxy = np.hstack([pred_XOxy,4*np.ones_like(pred_XOxy)])
    YOxy_meta = {'output_index':pred_XOxy[:,1].astype(int)}

    pred_YOxy = model.predict(pred_XOxy,Y_metadata=YOxy_meta)


    cseSal = 0
    cse_normSal = 0
    truthSal = dat[20:30,3]
    mean_truthSal = np.mean(truthSal)
    cnllSal = 0
    for i in range(10):
        pred_mean = meanSal + pred_YSal[0][i,0] * stdSal
        pred_var = pred_YSal[1][i,0] * stdSal**2
        cseSal += (pred_mean - truthSal[i])**2
        cse_normSal += (mean_truthSal - truthSal[i])**2
        cnllSal += 0.5 * ((pred_mean - truthSal[i])**2 / pred_var + np.log(2*np.pi*pred_var))

    cseOxy = 0
    cse_normOxy = 0
    truthOxy = dat[30:40,4]
    mean_truthOxy = np.mean(truthOxy)
    cnllOxy = 0
    for i in range(10):
        pred_mean = meanOxy + pred_YOxy[0][i,0] * stdOxy
        pred_var = pred_YOxy[1][i,0] * stdOxy**2
        cseOxy += (pred_mean - truthOxy[i])**2
        cse_normOxy += (mean_truthOxy - truthOxy[i])**2
        cnllOxy += 0.5 * ((pred_mean - truthOxy[i])**2 / pred_var + np.log(2*np.pi*pred_var))

    
    print("smseSal = {}".format(cseSal/cse_normSal))
    print("smseOxy = {}".format(cseOxy/cse_normOxy))
    print("average = {}".format((cseSal/cse_normSal + cseOxy/cse_normOxy)/2))
    print("mnllSal = {}".format(cnllSal/10))
    print("mnllOxy = {}".format(cnllOxy/10))
    print("average = {}".format((cnllSal/10 + cnllOxy/10)/2))


# Results:

# Trial 1:
# smseSal = 0.10239271002717211
# smseOxy = 0.04883199618682176
# average = 0.07561235310699693
# mnllSal = 3.1685164328490805
# mnllOxy = 2.127499480800946
# average = 2.648007956825013

# Trial 2:
# smseSal = 0.09027465247006487
# smseOxy = 0.2131194545074291
# average = 0.15169705348874699
# mnllSal = 2.226509518563331
# mnllOxy = 3.8774681000140054
# average = 3.0519888092886682

# Trial 3:
# smseSal = 0.07340337174621854
# smseOxy = 0.08262265987021256
# average = 0.07801301580821554
# mnllSal = 1.688124566168296
# mnllOxy = 2.578512673378579
# average = 2.1333186197734375

# Trial 4:
# smseSal = 0.0958180189204561
# smseOxy = 0.16720579983831446
# average = 0.13151190937938528
# mnllSal = 2.466207976952092
# mnllOxy = 3.530483346035022
# average = 2.998345661493557

# Trial 5:
# smseSal = 0.07097851158787918
# smseOxy = 0.0999299683436154
# average = 0.08545423996574729
# mnllSal = 1.3707329691067336
# mnllOxy = 3.0784285430652427
# average = 2.224580756085988

# Trial 6:
# smseSal = 0.07981208236541011
# smseOxy = 0.016262980694810875
# average = 0.04803753153011049
# mnllSal = 1.525926008886414
# mnllOxy = 1.5212872080292368
# average = 1.5236066084578255

# Trial 7:
# smseSal = 0.10990880933000702
# smseOxy = 0.16516706269958442
# average = 0.13753793601479572
# mnllSal = 3.34300651774684
# mnllOxy = 3.7230054549764127
# average = 3.5330059863616263

# Trial 8:
# smseSal = 0.09148169287083106
# smseOxy = 0.14412654264246946
# average = 0.11780411775665026
# mnllSal = 2.1971135055129114
# mnllOxy = 3.219618636827792
# average = 2.7083660711703517

# Trial 9:
# smseSal = 0.13458751481890768
# smseOxy = 0.18014223057216253
# average = 0.15736487269553512
# mnllSal = 3.852268557533268
# mnllOxy = 4.079252498486456
# average = 3.965760528009862

# Trial 10:
# smseSal = 0.08549209764236905
# smseOxy = 0.18066514621793375
# average = 0.1330786219301514
# mnllSal = 1.9544891570549534
# mnllOxy = 3.5644259909227856
# average = 2.7594575739888696


# Averages:
# smseSal = 0.09341494617793158 +/- 0.005673506840221591
# smseOxy = 0.12980738415733545 +/- 0.019360630064543354
# average = 0.11161116516763352 +/- 0.011143726200627247
# mnllSal = 2.379289521037392 +/- 0.2487762328718216
# mnllOxy = 3.1299981932536474 +/- 0.24634022636928674
# average = 2.7546438571455196 +/- 0.21015366888516424



