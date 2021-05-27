import GPy
import numpy as np


# Before running this test, ensure jura_sample.dat and jura_validation.dat,
# downloaded from https://sites.google.com/site/goovaertspierre/pierregoovaertswebsite/download/jura-data are in this folder.

# Run this test from the command line as
# `python3 slfm_experiment.py`


def make_dat():
    dat = np.zeros((359,5))

    f = open('jura_sample.dat')
    for i in range(13):
        f.readline()
    for i in range(259):
        s = f.readline()
        vals = s.split()
        dat[i,0] = vals[0]
        dat[i,1] = vals[1]
        dat[i,2] = vals[4]
        dat[i,3] = vals[8]
        dat[i,4] = vals[10]
    f.close()

    f = open('jura_validation.dat')
    f.readline()
    for i in range(100):
        s = f.readline()
        vals = s.split()
        dat[259+i,0] = vals[0]
        dat[259+i,1] = vals[1]
        dat[259+i,2] = vals[4]
        dat[259+i,3] = vals[8]
        dat[259+i,4] = vals[10]

    return dat
        

def make_X(train,j):
    X = np.zeros((251,2))
    Y = np.zeros(251)
    ind = 0
    for i in range(0,251):
        if not np.isnan(train[i,j]):
            X[ind,0] = train[i,0]
            X[ind,1] = train[i,1]
            Y[ind] = train[i,j]
            ind += 1
    return (X[0:ind,:],Y[0:ind][:,None])


if __name__ == "__main__":

    train = make_dat()
    test = make_dat()

    # Log transform
    for i in range(359):
        train[i,2] = np.log(train[i,2])
        train[i,3] = np.log(train[i,3])
        train[i,4] = np.log(train[i,4])
        
    # Remove final 100 Cd observations
    for i in range(259,359):
        train[i,2] = None


    K1 = GPy.kern.RBF(2)
    K2 = GPy.kern.RBF(2)
    K3 = GPy.kern.RBF(2)
    LCM = GPy.util.multioutput.LCM(input_dim=2,num_outputs=3,kernels_list=[K1,K2])

    XCd,YCd = make_X(train,2)
    XNi,YNi = make_X(train,3)
    XZn,YZn = make_X(train,4)

    # Normalize
    meanCd = np.mean(YCd)
    meanNi = np.mean(YNi)
    meanZn = np.mean(YZn)
    stdCd = np.std(YCd)
    stdNi = np.std(YNi)
    stdZn = np.std(YZn)

    YCd = (YCd - meanCd)/stdCd
    YNi = (YNi - meanNi)/stdNi
    YZn = (YZn - meanZn)/stdZn

    model = GPy.models.GPCoregionalizedRegression(X_list=[XCd,XNi,XZn],Y_list=[YCd,YNi,YZn],kernel=LCM)

    model.optimize('lbfgs',messages=True)

    pred_XCd = test[259:,0:2]
    pred_XCd = np.hstack([pred_XCd,0*np.ones((100,1))])
    YCd_meta = {'output_index':pred_XCd[:,2].astype(int)}
    pred_YCd = model.predict(pred_XCd,Y_metadata=YCd_meta)

    cae = 0
    cnll = 0
    truth = test[259:,2]
    for i in range(100):
        var = pred_YCd[1][i,0]
        cae += abs(np.exp(meanCd + pred_YCd[0][i,0] * stdCd) - truth[i])
        # Add NLL using a log-normal distribution
        cnll += (np.log(truth[i]) - meanCd)**2 / (2*var) + np.log(truth[i] * np.sqrt(var*2*np.pi))

    print("mae = {}".format(cae/100))
    print("mnll = {}".format(cnll/100))


# Results:

# Trial 1:
# mae = 0.5352407251213234
# mnll = 1.0506601767791481

# Trial 2:
# mae = 0.5352598816199191
# mnll = 1.0506509974894762

# Trial 3:
# mae = 0.5353327568276105
# mnll = 1.0505658466071517

# Trial 4:
# mae = 0.535231246232888
# mnll = 1.0506869254792597

# Trial 5:
# mae = 0.5352085338426449
# mnll = 1.0506914972603036

# Trial 6:
# mae = 0.5352138191383873
# mnll = 1.050647453861118

# Trial 7:
# mae = 0.5352436777492706
# mnll = 1.0506426504512771

# Trial 8:
# mae = 0.5352614795316198
# mnll = 1.0506771020200099

# Trial 9:
# mae = 0.5352367842584594
# mnll = 1.050725737157405

# Trial 10:
# mae = 0.5352614194655798
# mnll = 1.050653806982885


# Average mae = 0.5352490323787703 +/- 1.0430728952103705e-05
# Average mnll = 1.0250171798723584 +/- 6.1263417762416196e-06






    
