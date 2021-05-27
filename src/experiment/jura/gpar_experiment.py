# Use of GPAR is adapted from code snippets in the GPAR repository.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wbml.plot
import wbml.metric
from wbml.data.jura import load
from wbml.experiment import WorkingDirectory
from numpy import linspace
import random

from gpar import GPARRegressor, log_transform


# Run this test from the command line as
# `python3 gpar_experiment.py`


def inputs(df):
    return df.reset_index()[["x", "y"]].to_numpy()


if __name__ == "__main__":

    train, test = load()

    random.seed()
    
    # Fit and predict GPAR.
    model = GPARRegressor(
        scale=10.0,
        linear=False,
        nonlinear=True,
        nonlinear_scale=1.0,
        noise=0.1,
        impute=True,
        replace=True,
        normalise_y=True,
        transform_y=log_transform,
    )
    model.fit(inputs(train), train.to_numpy(), fix=False)
    means, lowers, uppers = model.predict(
        inputs(test), num_samples=200, credible_bounds=True, latent=False)
    means = pd.DataFrame(means, index=test.index, columns=train.columns)

    # uppers - lowers gives 95% confidence bound = 2*1.96 std
    std = (uppers - lowers)/(2*1.96)
    std = std[:,2]
    
    nll = (np.log(test["Cd"]) - means["Cd"])**2/(2*std**2) + np.log(test["Cd"] + std*np.sqrt(2*np.pi))

    print("mae = {}".format(wbml.metric.mae(means,test)["Cd"]))
    print("mnll = {}".format(nll.sum(axis=0)/100))


# Results:

# These results are taken using the option latent=True, since this was used in the GPAR paper and has better error performance. But this doesn't fully account for the variance, so LL is not good.

# Trial 1:
# mae = 0.39895354406468725
# mnll = 159.93358343044935

# Trial 2:
# mae = 0.4017492659286033
# mnll = 160.02450169247356

# Trial 3:
# mae = 0.3998050859203357
# mnll = 154.6038418949141

# Trial 4:
# mae = 0.4012336362418833
# mnll = 149.8658403919946

# Trial 5:
# mae = 0.3987225832324573
# mnll = 147.90553665546767

# Trial 6:
# mae = 0.39963426444195954
# mnll = 150.302476243423

# Trial 7:
# mae = 0.3993096034187578
# mnll = 158.78818539461014

# Trial 8:
# mae = 0.40095856704651356
# mnll = 170.62207044906722

# Trial 9:
# mae = 0.40058867594502423
# mnll = 153.14840112643367

# Trial 10:
# mae = 0.39763878117593987
# mnll = 161.87719804026895


# Average mae = 0.3998594007416162 +/- 0.0003821010036167846
# Average mnll = 156.70716353191023 +/- 2.069976256299762


# These results use latent=False, which gives worse mean errors but more accurate log likelihoods.

# Trial 1:
# mae = 0.4462947423253649
# mnll = 4.882040966888377

# Trial 2:
# mae = 0.4434608424789964
# mnll = 5.012522913783385

# Trial 3:
# mae = 0.43692505619180666
# mnll = 4.873877154458154

# Trial 4:
# mae = 0.4406903399788302
# mnll = 4.979974433150755

# Trial 5:
# mae = 0.43486862466884296
# mnll = 5.212932790271745

# Trial 6:
# mae = 0.43005633828206163
# mnll = 5.048665925940651

# Trial 7:
# mae = 0.444265861610267
# mnll = 5.166567487851947

# Trial 8:
# mae = 0.44207333466497367
# mnll = 4.80134087045241

# Trial 9:
# mae = 0.42984611930589817
# mnll = 5.354779867192507

# Trial 10:
# mae = 0.44306530567324315
# mnll = 4.873237160339039


# Average mae = 0.4391546565180285 +/- 0.0017736631849744437
# Average mnll = 5.020593957032896 +/- 0.05319723081714522










