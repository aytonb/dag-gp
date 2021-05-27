# This file modified from https://github.com/wesselb/gpar/blob/master/examples/paper/exchange.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wbml.plot
import wbml.metric
from wbml.data.exchange import load
from wbml.experiment import WorkingDirectory
from numpy import linspace

from gpar import GPARRegressor


# Run this file from the command line as
# `python3 gpar_experiment.py`


if __name__ == "__main__":

    _, train, test = load(nguyen=True)

    # Refill JPY
    train.iloc[99:150,11] = test.iloc[50:101,1]
    # Redefine test
    test = test[['USD/CAD','USD/AUD']]
    test.insert(1, 'USD/HKD', None, True)
    # Move HKD values to test
    test.iloc[50:101,1] = train.iloc[99:150,3]
    train.iloc[99:150,3] = None
    # Greedy ordering
    # EUR: -199.305
    # CHF: -277.128
    # XAU: -193.843
    # NZD: -102.592
    # XAG: -5.761
    # AUD: -233.794
    # CAD: -235.051
    # HKD: -73.231
    train = train[['USD/EUR','USD/CHF','USD/XAU','USD/NZD','USD/XAG','USD/AUD','USD/CAD','USD/HKD']]
   
    x = np.array(train.index)
    y = np.array(train)

    # Fit and predict GPAR.
    model = GPARRegressor(
        scale=0.1,
        linear=True,
        linear_scale=10.0,
        nonlinear=True,
        nonlinear_scale=1.0,
        rq=True,
        noise=0.01,
        impute=True,
        replace=False,
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
    std = std[:,5:]
    nll = 0.5 * ((pred[["USD/AUD","USD/CAD","USD/HKD"]] - test)**2 / (std**2) + np.log(2*np.pi * (std**2)))
    nll = nll.mean()
    

    # Report average SMSE.
    print("smseCAD = {}".format(smse["USD/CAD"]))
    print("smseHKD = {}".format(smse["USD/HKD"]))
    print("smseAUD = {}".format(smse["USD/AUD"]))
    print("average = {}".format(smse.mean()))
    print("mnllCAD = {}".format(nll["USD/CAD"]))
    print("mnllHKD = {}".format(nll["USD/HKD"]))
    print("mnllAUD = {}".format(nll["USD/AUD"]))
    print("average = {}".format(nll.mean()))


    # xx = np.arange(1,365,0.2)/365
    # means_full, lowers_full, uppers_full = model.predict(
    #     xx+2007, num_samples=200, credible_bounds=True, latent=False
    # )

    # out = np.concatenate((xx[:,None]*365,means_full[:,6][:,None],lowers_full[:,6][:,None],uppers_full[:,6][:,None],means_full[:,7][:,None],lowers_full[:,7][:,None],uppers_full[:,7][:,None],means_full[:,5][:,None],lowers_full[:,5][:,None],uppers_full[:,5][:,None]),axis=1)
    # np.savetxt('gpar_full_predict.txt',out)


# Results:

# Trial 1:
# smseCAD = 0.8924192481387317
# smseHKD = 1.5202713143451914
# smseAUD = 0.04052125968210738
# average = 0.8177372740553435
# mnllCAD = -2.6312633937793235
# mnllHKD = -7.591375238007971
# mnllAUD = -3.477465623779286
# average = -4.566701418522193

# Trial 2:
# smseCAD = 0.8393832040782581
# smseHKD = 1.072138122448712
# smseAUD = 0.042428868436154045
# average = 0.6513167316543748
# mnllCAD = -2.6547395206801765
# mnllHKD = -7.628576501864071
# mnllAUD = -3.457378119881214
# average = -4.580231380808487

# Trial 3:
# smseCAD = 0.9522903691655452
# smseHKD = 1.3932130894860661
# smseAUD = 0.042365717558752895
# average = 0.7959563920701215
# mnllCAD = -2.566647033617238
# mnllHKD = -7.566224326963689
# mnllAUD = -3.486420249208255
# average = -4.539763869929728

# Trial 4:
# smseCAD = 0.9094003220430708
# smseHKD = 1.1041980795170319
# smseAUD = 0.035549071234763446
# average = 0.6830491575982887
# mnllCAD = -2.6204493382375644
# mnllHKD = -7.605069772377577
# mnllAUD = -3.4949533789753517
# average = -4.5734908298634975

# Trial 5:
# smseCAD = 0.9635104312832301
# smseHKD = 1.793330351144921
# smseAUD = 0.044438470435958076
# average = 0.9337597509547031
# mnllCAD = -2.5516206927061402
# mnllHKD = -7.54239709629006
# mnllAUD = -3.453857911074903
# average = -4.515958566690368

# Trial 6:
# smseCAD = 1.1034530945328238
# smseHKD = 1.2482527167253425
# smseAUD = 0.0371973702188054
# average = 0.7963010604923239
# mnllCAD = -2.3970763173537306
# mnllHKD = -7.580133887036692
# mnllAUD = -3.481624500590774
# average = -4.4862782349937325

# Trial 7:
# smseCAD = 0.9237817032014144
# smseHKD = 1.1267117040138246
# smseAUD = 0.04214603853817662
# average = 0.6975464819178052
# mnllCAD = -2.582606946526128
# mnllHKD = -7.59049132161918
# mnllAUD = -3.4635732689735907
# average = -4.545557179039633

# Trial 8:
# smseCAD = 1.0052189309483817
# smseHKD = 1.3511333193120156
# smseAUD = 0.040881128714705804
# average = 0.7990777929917009
# mnllCAD = -2.563269695100615
# mnllHKD = -7.624138431246215
# mnllAUD = -3.491310511215195
# average = -4.559572879187342

# Trial 9:
# smseCAD = 0.9931175994918832
# smseHKD = 1.694061227105119
# smseAUD = 0.04545482710919998
# average = 0.910877884568734
# mnllCAD = -2.5519237699058586
# mnllHKD = -7.574983581649223
# mnllAUD = -3.4641135410416815
# average = -4.530340297532255

# Trial 10:
# smseCAD = 0.9431295902711575
# smseHKD = 1.4936531693168238
# smseAUD = 0.038096897517286094
# average = 0.8249598857017558
# mnllCAD = -2.5619108855115615
# mnllHKD = -7.554222185967194
# mnllAUD = -3.5406084554202324
# average = -4.552247175632996


# Averages:
# smseCAD = 0.9525704493154497 +/- 0.0215667265439689
# smseHKD = 1.379696309341505 +/- 0.07452747206937331
# smseAUD = 0.04090796494459098 +/- 0.0009468265793839075
# average = 0.7910582412005152 +/- 0.02768310913155223
# mnllCAD = -2.5681507593418336 +/- 0.021045898867343496
# mnllHKD = -7.585761234302187 +/- 0.008448279354779585
# mnllAUD = -3.4811305560160477 +/- 0.007602494008494845
# average = -4.545014183220023 +/- 0.00855751273084229




