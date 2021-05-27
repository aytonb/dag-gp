# Use of GPAR is adapted from code snippets in the GPAR repository.

import numpy as np
import pandas as pd
import wbml.metric
from numpy import linspace

from gpar import GPARRegressor

# Run this file from the command line. Download andro.arff into
# src/experiment/andro and cd into the folder. Then run
# `python3 gpar_experiment.py`.

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

    test_dat = np.copy(dat)
                
    train = pd.DataFrame(dat, index=np.linspace(0,53,54), columns=['Temp','PH','Cond','Sal','Oxy','Turb'])

    for i in range(20,30):
        train.loc[i,'Sal'] = None
    for i in range(30,40):
        train.loc[i,'Oxy'] = None
    

    test = pd.DataFrame(test_dat[20:40,(3,4)], index=np.linspace(20,39,20), columns=['Sal','Oxy'])


    for i in range(20,40):
        if not (i >= 20 and i <= 29):
            test.loc[i,'Sal'] = None
        if not (i >= 30 and i <= 39):
            test.loc[i,'Oxy'] = None

    # Greedy ordering for GPAR-NL (best)
    # Temp: 1.960
    # PH: -44.387
    # Turb: 31.549
    # Cond: 27.862
    # Oxy: -3.926
    # Sal: 1.548

    train = train[['Temp','PH','Turb','Cond','Oxy','Sal']]
    
   
    x = np.array(train.index)
    y = np.array(train)

    # Fit and predict GPAR.
    model = GPARRegressor(
        scale=1.0,
        linear=False,
        linear_scale=1.0,
        nonlinear=True,
        nonlinear_scale=1.0,
        rq=False,
        noise=0.1,
        impute=True,
        replace=True,
        normalise_y=True,
    )
    model.fit(x, y)
    means, lowers, uppers = model.predict(
        x, num_samples=200, credible_bounds=True, latent=False
    )

    pred = pd.DataFrame(means, index=train.index, columns=train.columns)
    smse = wbml.metric.smse(pred, test)

    # uppers - lowers fives 95% confidence bound = 2*1.96 std
    std = (uppers - lowers)/(2*1.96)
    std = std[:,4:]
    nll = 0.5 * ((pred[["Oxy","Sal"]] - test)**2 / (std**2) + np.log(2*np.pi * (std**2)))
    nll = nll.mean()

    
    # Report average SMSE
    print("smseSal = {}".format(smse["Sal"]))
    print("smseOxy = {}".format(smse["Oxy"]))
    print("average = {}".format(smse.mean()))
    print("mnllSal = {}".format(nll["Sal"]))
    print("mnllOxy = {}".format(nll["Oxy"]))
    print("average = {}".format(nll.mean()))


# Results:

# Trial 1:
# smseSal = 0.2013844783691025
# smseOxy = 0.4521982569261676
# average = 0.32679136764763506
# mnllSal = 1.0512320742876358
# mnllOxy = 3.189772530140396
# average = 2.1205023022140157

# Trial 2:
# smseSal = 0.21975603529061785
# smseOxy = 0.463314732949377
# average = 0.3415353841199974
# mnllSal = 1.1595694979168867
# mnllOxy = 3.1602429826064444
# average = 2.1599062402616656

# Trial 3:
# smseSal = 0.20246394829846978
# smseOxy = 0.46855519394529305
# average = 0.3355095711218814
# mnllSal = 1.004956780061132
# mnllOxy = 3.195639394850115
# average = 2.1002980874556236

# Trial 4:
# smseSal = 0.2202176009937325
# smseOxy = 0.4654131130313983
# average = 0.3428153570125654
# mnllSal = 1.1544523036188914
# mnllOxy = 3.238773514165177
# average = 2.196612908892034

# Trial 5:
# smseSal = 0.20741236196853058
# smseOxy = 0.47637851399950215
# average = 0.34189543798401634
# mnllSal = 1.0358630609827064
# mnllOxy = 3.190237816304466
# average = 2.113050438643586

# Trial 6:
# smseSal = 0.21294015829955332
# smseOxy = 0.4522245946183183
# average = 0.33258237645893585
# mnllSal = 1.0787847961602341
# mnllOxy = 3.219131789283167
# average = 2.1489582927217006

# Trial 7:
# smseSal = 0.21178527058318847
# smseOxy = 0.4661576120302574
# average = 0.33897144130672296
# mnllSal = 1.0516551457190153
# mnllOxy = 3.194055902805741
# average = 2.122855524262378

# Trial 8:
# smseSal = 0.21120985047245705
# smseOxy = 0.4575764758373672
# average = 0.33439316315491213
# mnllSal = 1.1210310252740332
# mnllOxy = 3.2179405134508086
# average = 2.1694857693624208

# Trial 9:
# smseSal = 0.19611291124930613
# smseOxy = 0.45072822758659803
# average = 0.32342056941795205
# mnllSal = 0.969984966529602
# mnllOxy = 3.2171502659943165
# average = 2.0935676162619594

# Trial 10:
# smseSal = 0.20326408027500928
# smseOxy = 0.4730295841507609
# average = 0.3381468322128851
# mnllSal = 1.0420673522422157
# mnllOxy = 3.1548807960223266
# average = 2.0984740741322714


# Averages:
# smseSal = 0.20865466957999673 +/- 0.002388964048766852
# smseOxy = 0.46255763050750404 +/- 0.002715089647341296
# average = 0.33560615004375033 +/- 0.0019526303056812222
# mnllSal = 1.0669597002792353 +/- 0.01860423339303791
# mnllOxy = 3.1977825505622954 +/- 0.0079415647872017
# average = 2.132371125420766 +/- 0.010404771476357474
