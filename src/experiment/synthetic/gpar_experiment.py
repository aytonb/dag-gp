from matplotlib import pyplot
import numpy as np
import pandas as pd
import wbml.plot
import wbml.metric
from wbml.data.exchange import load
from wbml.experiment import WorkingDirectory
from numpy import linspace
import random

from gpar import GPARRegressor

# Run this test from the command line as
# `python3 gpar_experiment.py`

def make_dat():
    x = np.linspace(0,10,41)
    y1 = np.exp(-(x-2)**2)
    y2 = 5*np.exp(-0.4*(x-2.5)**2) + 3*np.exp(-4*(x-7)**2) + 0.01*x**3 + 1.1*np.exp(-x**2)
    y3 = 2*y1 + 0.8*np.log(x+0.25)

    return np.hstack((x[:,None],y1[:,None],y2[:,None],y3[:,None]))

if __name__ == "__main__":   
    dat = make_dat()
    test_dat = make_dat()
                
    train = pd.DataFrame(dat[:,1:], index=np.arange(0,10.25,0.25), columns=['0','1','2'])

    for i in range(24,33):
        train.iloc[i,2] = None

    test = pd.DataFrame(test_dat[24:33,(3)], index=np.arange(6,8.25,0.25), columns=['2'])

    # Greedy ordering optimized
    train = train[['0','1','2']]

    random.seed()
   
    x = np.array(train.index)
    y = np.array(train)

    # Fit and predict GPAR.
    model = GPARRegressor(
        scale=1.0,
        linear=True,
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

    for i in range(9):
        print("pred = {}, truth = {}".format(means[24+i,2],test.iloc[i,0]))

    pred = pd.DataFrame(means, index=train.index, columns=train.columns)
    cae2 = (abs(pred - test)).sum(axis=0)
    cse2 = ((pred - test) ** 2).sum(axis=0)

    print("cae2 = {}".format(cae2.iloc[2]))
    print("cse2 = {}".format(cse2.iloc[2]))

    pyplot.figure()
    pyplot.plot(x,means[:,0])
    pyplot.plot(x,means[:,1])
    pyplot.plot(x,means[:,2])

    pyplot.figure()
    pyplot.plot(x,means[:,2]-test_dat[:,3])

    pyplot.show()

    # xx = np.arange(0,10.05,0.05)
    # means_full, lowers_full, uppers_full = model.predict(
    #     xx, num_samples=5000, credible_bounds=True, latent=False
    # )

    # out = np.concatenate((xx[:,None],means_full[:,0][:,None],lowers_full[:,0][:,None],uppers_full[:,0][:,None],means_full[:,1][:,None],lowers_full[:,1][:,None],uppers_full[:,1][:,None],means_full[:,2][:,None],lowers_full[:,2][:,None],uppers_full[:,2][:,None]),axis=1)
    # np.savetxt('gpar_full_predict_2.txt',out)


# Results:

# Trial 1:
# pred = 1.4700193944090294, truth = 1.4660653960689978
# pred = 1.533237018874597, truth = 1.497441770135757
# pred = 1.6729872122951992, truth = 1.527634007118007
# pred = 1.8820825049816705, truth = 1.5567281195621288
# pred = 2.009306395101804, truth = 1.5848011751210425
# pred = 1.9398622549158835, truth = 1.6119224164359536
# pred = 1.7869899269609268, truth = 1.63815427469235
# pred = 1.7015256839537147, truth = 1.6635532333438774
# pred = 1.6926538630195884, truth = 1.688170560277272
# cae2 = 1.454193
# cse2 = 0.439643

# Trial 2:
# pred = 1.468977249858532, truth = 1.4660653960689978
# pred = 1.5280153798016511, truth = 1.497441770135757
# pred = 1.6569569350441737, truth = 1.527634007118007
# pred = 1.8500633441928438, truth = 1.5567281195621288
# pred = 1.9691783177550994, truth = 1.5848011751210425
# pred = 1.9092020493057311, truth = 1.6119224164359536
# pred = 1.772592260193758, truth = 1.63815427469235
# pred = 1.6975672234823185, truth = 1.6635532333438774
# pred = 1.6920859241110984, truth = 1.688170560277272
# cae2 = 1.31016773098982
# cse2 = 0.3590800196405205

# Trial 3:
# pred = 1.4679945273146844, truth = 1.4660653960689978
# pred = 1.5225480688178377, truth = 1.497441770135757
# pred = 1.6386595771810974, truth = 1.527634007118007
# pred = 1.811715429192074, truth = 1.5567281195621288
# pred = 1.920594767614714, truth = 1.5848011751210425
# pred = 1.8717261750565009, truth = 1.6119224164359536
# pred = 1.7552121106465424, truth = 1.63815427469235
# pred = 1.692917907358032, truth = 1.6635532333438774
# pred = 1.69142015272018, truth = 1.688170560277272
# cae2 = 1.1383177631462766
# cse2 = 0.27280996370347943

# Trial 4:
# pred = 1.4692819010802554, truth = 1.4660653960689978
# pred = 1.5294347028058943, truth = 1.497441770135757
# pred = 1.6614131818567068, truth = 1.527634007118007
# pred = 1.8592790974897133, truth = 1.5567281195621288
# pred = 1.980686562126965, truth = 1.5848011751210425
# pred = 1.9171919610278738, truth = 1.6119224164359536
# pred = 1.7755066026598385, truth = 1.63815427469235
# pred = 1.6978974364051174, truth = 1.6635532333438774
# pred = 1.6920452721841202, truth = 1.688170560277272
# cae2 = 1.3482657648810985
# cse2 = 0.38044278965857736

# Trial 5:
# pred = 1.4694351268222283, truth = 1.4660653960689978
# pred = 1.5317179733192847, truth = 1.497441770135757
# pred = 1.6697758765249175, truth = 1.527634007118007
# pred = 1.8765045814607666, truth = 1.5567281195621288
# pred = 2.00253573151516, truth = 1.5848011751210425
# pred = 1.9343290435837626, truth = 1.6119224164359536
# pred = 1.783686564606786, truth = 1.63815427469235
# pred = 1.7000697492566694, truth = 1.6635532333438774
# pred = 1.692203775255735, truth = 1.688170560277272
# cae2 = 1.4257874695899242
# cse2 = 0.4246250728127575

# Trial 6:
# pred = 1.4678757980381827, truth = 1.4660653960689978
# pred = 1.5229734830405783, truth = 1.497441770135757
# pred = 1.6428717324187345, truth = 1.527634007118007
# pred = 1.8237515787334675, truth = 1.5567281195621288
# pred = 1.9365354146188765, truth = 1.5848011751210425
# pred = 1.8819973042398983, truth = 1.6119224164359536
# pred = 1.7574324733453934, truth = 1.63815427469235
# pred = 1.6919601047149186, truth = 1.6635532333438774
# pred = 1.6909488772333685, truth = 1.688170560277272
# cae2 = 1.1818758136280323
# cse2 = 0.29693578531700504

# Trial 7:
# pred = 1.469710146598756, truth = 1.4660653960689978
# pred = 1.5315758160337558, truth = 1.497441770135757
# pred = 1.6666135077738236, truth = 1.527634007118007
# pred = 1.8679555366950538, truth = 1.5567281195621288
# pred = 1.9915140843006387, truth = 1.5848011751210425
# pred = 1.9274431210621576, truth = 1.6119224164359536
# pred = 1.782651993507583, truth = 1.63815427469235
# pred = 1.7012529381377575, truth = 1.6635532333438774
# pred = 1.69292938335725, truth = 1.688170560277272
# cae2 = 1.39717557471139
# cse2 = 0.40464843449616783

# Trial 8:
# pred = 1.4690593766803812, truth = 1.4660653960689978
# pred = 1.5319789716270593, truth = 1.497441770135757
# pred = 1.6769722924588757, truth = 1.527634007118007
# pred = 1.8983859460878072, truth = 1.5567281195621288
# pred = 2.033202353367314, truth = 1.5848011751210425
# pred = 1.956803813564768, truth = 1.6119224164359536
# pred = 1.7918187844955873, truth = 1.63815427469235
# pred = 1.7010717039749974, truth = 1.6635532333438774
# pred = 1.6921898398442647, truth = 1.688170560277272
# cae2 = 1.5170121293456686
# cse2 = 0.48527714265952593

# Trial 9:
# pred = 1.4697421407975242, truth = 1.4660653960689978
# pred = 1.5327443171087933, truth = 1.497441770135757
# pred = 1.672785570452218, truth = 1.527634007118007
# pred = 1.8827425700662324, truth = 1.5567281195621288
# pred = 2.0103043822541644, truth = 1.5848011751210425
# pred = 1.9401601230533536, truth = 1.6119224164359536
# pred = 1.7863079072961239, truth = 1.63815427469235
# pred = 1.7010882165589458, truth = 1.6635532333438774
# pred = 1.6926110715947988, truth = 1.688170560277272
# cae2 = 1.4540153464267684
# cse2 = 0.44078524983587375

# Trial 10:
# pred = 1.468706828024429, truth = 1.4660653960689978
# pred = 1.529460371178255, truth = 1.497441770135757
# pred = 1.6659432540767989, truth = 1.527634007118007
# pred = 1.8722283920843643, truth = 1.5567281195621288
# pred = 1.9989391716857166, truth = 1.5848011751210425
# pred = 1.9313850076394958, truth = 1.6119224164359536
# pred = 1.7815437343622704, truth = 1.63815427469235
# pred = 1.6992081805210577, truth = 1.6635532333438774
# pred = 1.69201656286123, truth = 1.688170560277272
# cae2 = 1.4049605496782318
# cse2 = 0.4151152692470054


# Average mae = 0.1514641238044135 +/- 0.004059537277935741

