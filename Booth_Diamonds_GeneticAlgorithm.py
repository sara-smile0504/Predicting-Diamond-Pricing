#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sara Foster
"""

from deap import creator, base, tools, algorithms

import random, sys, time, warnings
import pandas as pd
import numpy  as np
import matplotlib.pyplot               as plt
import statsmodels.api                 as sm
import statsmodels.tools.eval_measures as em
from AdvancedAnalytics.ReplaceImputeEncode import ReplaceImputeEncode, DT
from AdvancedAnalytics.Regression          import linreg, stepwise
from math                 import log, isfinite, sqrt, pi
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.metrics      import mean_squared_error, r2_score
from scipy.linalg         import qr_multiply, solve_triangular
              
def rngFit(z):
    r = maxFit(z) - minFit(z)
    return round(r, 3)

def avgFit(z):
    tot = 0.0
    cnt = 0
    for i in range(len(z)):
        if isfinite(z[i][0]):
            tot += z[i][0]
            cnt += 1
    if cnt>0:
        return round(tot/cnt, 4)
    else:
        return np.nan

def maxFit(z):
    maximum = 0
    for i in range(len(z)):
        if z[i][0] > maximum:
            maximum = z[i][0]
    return maximum

def minFit(z):
    minimum = np.inf
    for i in range(len(z)):
        if z[i][0] < minimum:
            minimum = z[i][0]
    return minimum

def cvFit(z):
    avg = avgFit(z)
    std = stdFit(z)
    if isfinite(avg):
        return round(100*std/avg, 3)
    else:
        return np.nan

def logMinFit(z):
    try:
        return round(log(minFit(z)), 6)
    except:
        return np.inf
    
def logMaxFit(z):
    try:
        return round(log(maxFit(z)), 6)
    except:
        return -np.inf

def stdFit(z):
    sum1  = 0.0
    sum2  = 0.0
    cnt   = 0
    for i in range(len(z)):
        if isfinite(z[i][0]):
            sum1 += z[i][0]
            sum2 += z[i][0] * z[i][0]
            cnt += 1
    if cnt < 2:
        return np.nan
    else:
        sumsq = (sum1*sum1)/cnt
        return round(sqrt((sum2 - sumsq)/(cnt-1)), 4)
def features_min(z):
    minimum = np.inf
    feature = np.inf
    for i in range(len(z)):
        if z[i][0] < minimum:
            minimum = z[i][0]
            feature = z[i][1]
        if z[i][0] == minimum and z[i][1] < feature:
            feature = z[i][1]
    return round(feature,0)

def features_max(z):
    maximum = -np.inf
    feature =  np.inf
    for i in range(len(z)):
        if z[i][0] > maximum:
            maximum = z[i][0]
            feature = z[i][1]
        if z[i][0] == maximum and z[i][1] < feature:
            feature = z[i][1]
    return round(feature,0)

def geneticAlgorithm(X, y, n_population, n_generation, method=None,
                     reg=None, goodFit=None,  calcModel=None,
                     n_int=None, n_nom=None, n_frac=None):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
    if method==None:
        method = 'random'
    if goodFit==None:
        goodFit='bic'
    if calcModel==None:
        calcModel='statsmodels'
    if type(y)==np.ndarray:
        nval = len(np.unique(y))
    else:
        nval = y.nunique()
    if reg==None:
        if nval > 20:
            reg = 'linear'
        else: 
            reg = 'logistic'
    if goodFit.lower()!='adjr2':
        opt = -1.0 # Minimize goodness of fit
    else:
        opt =  1.0 # Maximize goodness of fit
 # create individual fitness dictionary
    ifit = {}
    # create individual
    # Two weights for two optimization (goodness of fit, number of features)
    # A negative weight indicates minimize that function.
    # A positive weight indicates maximize that function.
    with warnings.catch_warnings():  
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        creator.create("FitnessMax", base.Fitness, weights=(opt, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("population_guess", initPopulation, list, 
                                                      creator.Individual)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                                     toolbox.attr_bool, n=len(X.columns))
    toolbox.register("population", tools.initRepeat, list,
                                                      toolbox.individual)
    if   reg.lower()=='logistic':
        toolbox.register("evaluate", evalFitnessLogistic, X=X, y=y, 
                         goodFit=goodFit, calcModel=calcModel, ifit=ifit)
    elif reg.lower()=='linear':
        toolbox.register("evaluate", evalFitnessLinear, X=X, y=y, 
                         goodFit=goodFit, calcModel=calcModel, ifit=ifit)
    else:
        raise ValueError("reg not set to 'linear' or 'logistic'")
        sys.exit()
    toolbox.register("mate",     tools.cxTwoPoint)
    toolbox.register("mutate",   tools.mutFlipBit, indpb=0.02)
    toolbox.register("select",   tools.selTournament, tournsize=7)

    if method=='random':
        pop   = toolbox.population(n_population)
    else:
        # initialize parameters
        # n_int Total number of interval features
        # n_nom List of number of dummy variables for each categorical var
        pop   = toolbox.population_guess(method, n_int, n_nom, n_frac)
        #n_population = len(pop)
    hof   = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    if goodFit.lower!='adjr2':
        stats.register("features", features_min)
    else:
        stats.register("features", features_max)
    stats.register("range",    rngFit)
    stats.register("min",      minFit)
    stats.register("avg",      avgFit)
    stats.register("max",      maxFit)
    if goodFit.lower()!='adjr2':
        stats.register("Ln(Fit)",  logMinFit)
    else:
        stats.register("Ln(Fit)",  logMaxFit)
        

    # genetic algorithm
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.5,
                                   ngen=n_generation, stats=stats, 
                                   halloffame=hof, verbose=True)

    # return hall of fame
    return hof, logbook

def evalFitnessLinear(individual, X, y, goodFit, calcModel, ifit):
    # returns (goodness of fit, number of features)
    cols  = [index for index in range(len(individual)) 
            if individual[index] == 1 ]# get features subset, 
                                       # drop features with cols[i] != 1
    if type(X)==np.ndarray:
        X_selected = X[:, cols]
    else:
        X_selected = X.iloc[:, cols]
    
    features = X_selected.shape[1]
    n = X_selected.shape[0]
    p = features
    k = features + 2 # 2 for intercept and variance
    ind = ""     
    for i in range(len(individual)):
        if individual[i] == 0:
            ind += '0'
        else:
            ind += '1'
    try:
        fit = ifit[ind]
        return(fit, features)
    except:
        pass
    goodFit   = goodFit.lower()
    calcModel = calcModel.lower()
    if   k > n+2 and goodFit=='bic':
        return (np.inf, features)
    elif k > n+2 and goodFit=='adjr2':
        return (0, features)
    
    if calcModel == "qr_decomp":
        Xc     = sm.add_constant(X_selected)
        qty, r = qr_multiply(Xc, y)
        coef   = solve_triangular(r, qty)
        pred   = (Xc @ coef)
        resid  = pred - y
        ASE    = (resid @ resid) / n
        if ASE > 0:
            twoLL = n*(log(2*pi) + 1.0 + log(ASE))
            bic   = twoLL + log(n)*k
            aic   = twoLL + 2*k
            R2    = r2_score(y, pred)
            if R2 > 0.99999:
                bic = -np.inf
        else: 
            bic = -np.inf
            aic = -np.inf
            R2  = 1.0
            
        if goodFit == 'bic':
            return(bic, features)
        elif goodFit == 'aic':
            return(aic, features)
        else:
            adjr2 = 1.0-R2 
            adjr2 = ((n-1)/(n-p-1))*adjr2
            if adjr2 < 1.0:
                adjr2 = 1.0 - adjr2
                return(adjr2, features)
            else:
                return(0.0, features)

    elif calcModel== "statsmodels":
        Xc       = sm.add_constant(X_selected)
        model    = sm.OLS(y, Xc)
        results  = model.fit()
        parms    = np.ravel(results.params)
        if goodFit == "adjr2":
            pred  = model.predict(parms)
            R2    = r2_score(y, pred)
            adjr2 = 1.0-R2 
            adjr2 = ((n-1)/(n-p-1))*adjr2
            if adjr2 < 1.0:
                adjr2 = 1.0 - adjr2
                return(adjr2, features)
            else:
                return(0.0, features)
        else:
            loglike  = model.loglike(results.params)
            model_df = model.df_model + 2 #plus intercept and sigma
            nobs     = y.shape[0]
            if goodFit=='bic':
                bic  = em.bic(loglike, nobs, model_df)
                return(bic, features)
            else:
                aic  = em.aic(loglike, nobs, model_df)
                return(aic, features)
        
    elif calcModel=='sklearn':
        # sklearn linear regression does not handle no features
        if X_selected.shape[1]>0:
            lr   = LinearRegression().fit(X_selected,y)
            pred = lr.predict(X_selected)
        else:
            avg  = y.mean()
            pred = np.array([avg]*y.shape[0])
        ASE  = mean_squared_error(y,pred)
        if ASE > 0:
            twoLL = n*(log(2*pi) + 1.0 + log(ASE))
            bic   = twoLL + log(n)*k
            aic   = twoLL + 2*k
            R2 = r2_score(y, pred)
            if R2 > 0.99999:
                bic = -np.inf
        else: 
            R2  = r2_score(y, pred)
            bic = -np.inf
            aic = -np.inf
            
        if goodFit == 'bic':
            return(bic, features)
        elif goodFit=='aic':
            return(aic, features)
        else:
            adjr2 = 1.0-R2 
            adjr2 = ((n-1)/(n-p-1))*adjr2
            if adjr2 < 1.0:
                adjr2 = 1.0 - adjr2
                return(adjr2, features)
            else:
                return(0.0, features)
    else:
        raise ValueError("calcModel not 'statsmodels', 'sklearn', or 'QR_decomp'")
        sys.exit()
    
def evalFitnessLogistic(individual, X, y, goodFit, calcModel, ifit):
    # Number of categories in y
    if type(y)==np.ndarray:
        n_cat = len(np.unique(y))
    else:
        n_cat = y.nunique()
    # returns (goodness of fit, number of features)
    cols  = [index for index in range(len(individual)) 
            if individual[index] == 1 ]# get features subset, 
                                       # drop features with cols[i] != 1
    if type(X)==np.ndarray:
        X_selected = X[:, cols]
    else:
        X_selected = X.iloc[:, cols]
    
    features = X_selected.shape[1]
    n = X_selected.shape[0]
    p = features
    if n_cat <= 2:
        k = features + 2 #for intercept and varianc
    else:
        k = n_cat*(features + 1) + 1 # n_cat intercepts and +1 for variance
    ind = ""     
    for i in range(len(individual)):
        if individual[i] == 0:
            ind += '0'
        else:
            ind += '1'
    try:
        fit = ifit[ind]
        return(fit, features)
    except:
        pass
    goodFit   = goodFit.lower()
    calcModel = calcModel.lower()
    if   k > n+2 and goodFit=='bic':
        return (np.inf, features)
    elif k > n+2 and goodFit=='adjr2':
        return (0, features)

    if calcModel== "statsmodels":
        Xc = sm.add_constant(X_selected)
        try:
            model   = sm.Logit(y, Xc)
            results = model.fit(disp=False) 
        except:
            print("Singular Fit Encountered with", features, "features")
            if goodFit != 'adjr2':
                return(-np.inf, features)
            else:
                return(1.0, features)
        proba = model.predict(results.params)   
        if goodFit == "adjr2":
            R2    = r2_score(y, proba)
            adjr2 = 1.0-R2 
            adjr2 = ((n-1)/(n-p-1))*adjr2
            if adjr2 < 1.0:
                adjr2 = 1.0 - adjr2
                return(adjr2, features)
            else:
                return(0.0, features)
        else:
            ll = 0
            for i in range(n):
                if y[i] == 1:
                    d = log(proba[i])
                else:
                    d = log(1.0 - proba[i])
                ll += d
            if goodFit=='bic':
                bic  = em.bic(ll, n, k)
                return(bic, features)
            else:
                aic  = em.aic(ll, n, k)
                return(aic, features)
        
    elif calcModel=='sklearn':
        # sklearn linear regression does not handle no features
        if X_selected.shape[1]>0:
            if X_selected.shape[0]*X_selected.shape[1] > 100000:
                opt='saga'
            else:
                opt='lbfgs'
            lr    = LogisticRegression(penalty='none', solver=opt,
                                  tol=1e-4, max_iter=5000)
            lr    = lr.fit(X_selected, y)
            proba = lr.predict_proba(X_selected)
        else:
            proba = np.full((y.shape[0],2),0.5)
        ll = 0
        for i in range(y.shape[0]):
            if y[i] == 1:
                d = log(proba[i,1])
            else:
                d = log(proba[i,0])
            ll += d
        twoLL = -2.0*ll
        if goodFit == 'bic':
            bic   = twoLL + log(n)*k
            return(bic, features)
        elif goodFit=='aic':
            aic   = twoLL + 2*k
            return(aic, features)
        else:
            R2 = r2_score(y, proba[:,1])
            adjr2 = 1.0-R2 
            adjr2 = ((n-1)/(n-p-1))*adjr2
            if adjr2 < 1.0:
                adjr2 = 1.0 - adjr2
                return(adjr2, features)
            else:
                return(0.0, features)
    else:
        raise ValueError("calcModel not 'statsmodels', 'sklearn', or 'QR_decomp'")
        sys.exit()
    
def initPopulation(pcls, ind_init, method, 
                   n_int, n_nom, n_frac):
    #k = number of columns in X
    #k1= number of interval variables (first k1 columns)
    #k2= number of other columns in X
    if n_int==None:
        k1 = 0
    elif type(n_int)==int:
        k1 = n_int
    else:
        k1 = 0
        
    if n_nom==None:
        k2 = 0
    elif type(n_nom)==int:
        k2 = n_nom
    else:
        k2 = sum(n_nom)
    k = k1+k2
    # Initialize Null Case (no features)
    icls = [0]*k
    ind  = ind_init(icls)
    pcls = [ind]
    
    if method == 'star':
        # Add "All" one-feature selection (star points)
        for i in range(k):
            icls = [0]*k
            icls[i]  = 1
            ind = ind_init(icls)
            pcls.append(ind)
            
    return pcls

def findBest(hof, goodFit, X, y, top=None):
    #Find Best Individual in Hall of Fame
    print("Individuals in HoF: ", len(hof))
    if top==None:
        top=1
    goodFit = goodFit.lower()
    features = np.inf
    if goodFit=='bic' or goodFit=='aic':
        bestFit = np.inf
        for individual in hof:
            if(individual.fitness.values[0] < bestFit):
                bestFit = individual.fitness.values[0]
                _individual = individual
            if (sum(individual) < features and 
                individual.fitness.values[0] == bestFit):
                features = sum(individual)
                _individual = individual
    elif goodFit=='adjr2':
        bestFit = -np.inf
        for individual in hof:
            if(individual.fitness.values[0] > bestFit):
                bestFit = individual.fitness.values[0]
                _individual = individual
            if (sum(individual) < features and 
                individual.fitness.values[0] == bestFit):
                features = sum(individual)
                _individual = individual
    else:
        raise RuntimeError("goodFit invalid: "+goodFit)
        sys.exit()
    if type(X)==np.ndarray:
        z = np.ravel(_individual)
        z = z.nonzero()
        _individualHeader = z[0]
    else:
        _individualHeader = [list(X)[i] for i in range(len(_individual)) 
                        if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader

        
def plotGenerations(gen, lnbic, features):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("GA GENERATION", fontsize="x-large",fontweight="heavy")
    ax1.tick_params(axis='x', labelcolor="black", labelsize="x-large")
    ax1.tick_params(axis='y', labelcolor="green", labelsize="x-large")
    #ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax1.set_ylabel("Log(BIC)", fontsize="x-large", fontweight="heavy", 
                   color="green")
    ax1.set_facecolor((0.95,0.95,0.95))
    #ax1.grid(axis='x', linestyle='--', linewidth=1, color='gray')
    ax1.plot(gen, lnbic, 'go-', color="green", 
                         linewidth=2, markersize=10)
    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor="blue", labelsize="x-large")
    ax2.set_ylabel("Number of Features Selected", fontsize="x-large", 
                   fontweight="heavy", color="blue")
    #ax2.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
    #ax2.grid(axis='y', linestyle='--', linewidth=1, color='gray')
    ax2.plot(gen, features, 'bs-', color="blue", 
                         linewidth=2, markersize=10)
    plt.savefig("GA_Feature_Select.pdf")
    plt.show()
#*****************************************************************************
print("{:*>71s}".format('*'))
attribute_map = { 
    'obs': [DT.Ignore, ()], 
    'price': [DT.Interval, (300, 20000)], 
    'carat': [DT.Interval, (0.2, 5.5)], 
    'cut': [DT.Nominal, ('Fair', 'Good', 'Ideal', 'Premium', 'Very Good')], 
    'color': [DT.Nominal, ('D', 'E', 'F', 'G', 'H', 'I', 'J')], 
    'clarity': [DT.Nominal, ('I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2')], 
    'depth': [DT.Interval, (40, 80)], 
    'table': [DT.Interval, (40, 100)], 
    'x': [DT.Interval, (0, 11)], 
    'y': [DT.Interval, (0, 60)], 
    'z': [DT.Interval, (0, 32)]
    }

    
target = 'price'
df = pd.read_excel("diamonds_train.xlsx")
print("Read", df.shape[0], "observations with ", 
      df.shape[1], "attributes:\n")

rie = ReplaceImputeEncode(data_map=attribute_map, 
                          nominal_encoding='one-hot',
                          drop=False, display=True)
encoded_df = rie.fit_transform(df)

y = encoded_df[target] # The target is not scaled or imputed
X = encoded_df.drop(target, axis=1)

print("{:*>71s}".format('*'))
# apply genetic algorithm
# n_init:  set to the number of candidate interval and binary features
# n_nom:   set to a list of levels for each candidate nominal feature
#          if there are no candidate nominal features, set to an empty list []
n_int = 6       # number of interval and binary features (excludes target)
n_nom = [5, 7, 8] # 42 dummy features count each nominal variable
p     = n_int + sum(n_nom) # Total number of features 52

# modes:   the list of currently available statistical models
# fitness: the list of currently available fitness functions
# init:    the list of currently available initialization algorithms
#          each initialization algorithm can be used to initialize 
#          generation zero.  Select the one that produces a generation zero
#          closest to the imagined best number of features.  'star' starts 
#          with only one feature per individual.  'random' starts with a
#          larger number of features per individual, approximate half the
#          total number of candidates.
models     = [ 'sklearn', 'statsmodels', 'QR_decomp']
fitness    = ['bic', 'aic', 'AdjR2']
init       = ['star', 'random']
# Set calcModel, goodFit and initMethod to your choice for the statistical
#     model, the goodness of fit metric, and the initialization algorithm.
calcModel  = models [0]
goodFit    = fitness[0]
initMethod = init[0] #Initial generation has only 1 feature per individual.
             #Initial generation with 'random' has about 50% of all features.
# n_pop is the initial population size.  Subsequent generations will be near
#       this size.
# n_gen is the number of generations, each progressively better than the 
#       previous generation.  This needs to be large enough to all the 
#       search algorithm to identify the best feature selection.
# Note: This algorithm optimizes the fitness of the individual while 
#       minimizing the number of features selected for the model.
if initMethod=='star':
    n_pop = p+1
    n_gen =  50
else:
    n_pop = 100
    n_gen =  50

print("{:*>71s}".format('*'))
print("{:*>14s}     GA Selection using {:>5s} Fitness         {:*>11s}". 
      format('*', goodFit, '*'))
print("{:*>14s} {:>11s} Models and {:>6s} Initialization {:*>11s}". 
      format('*', calcModel, initMethod, '*'))
print(" ")
random.seed(12345)
start = time.time()
hof, logbook = geneticAlgorithm(X, y, n_pop, n_gen, method=initMethod,
                                reg='linear', goodFit=goodFit,
                                calcModel=calcModel, n_int=n_int, n_nom=n_nom)

gen, features, min_, avg_, max_, rng_, lnfit = logbook.select("gen",
                    "features", "min", "avg", "max", "range", "Ln(Fit)")
end = time.time()    
duration = end-start
print("GA Runtime ", duration, " sec.")

# Plot Fitness and Number of Features versus Generation
plotGenerations(gen, lnfit, features)

# select the best individual
fit, individual, header = findBest(hof, goodFit, X, y)
print("Best Fitness:", fit[0])
print("Number of Features Selecterd: ", len(header))
print("\nFeatures:", header)

Xc = sm.add_constant(X[header])
model = sm.OLS(y, Xc)
results = model.fit()
print(results.summary())
print(" ")
print("{:*>71s}".format('*'))
print("{:*>14s}     STEPWISE SELECTION    {:*>30s}". format('*', '*'))
print("{:*>71s}".format('*'))

sw       = stepwise(encoded_df, target, reg="linear", method="stepwise",
                    crit_in=0.1, crit_out=0.1, verbose=True)
selected = sw.fit_transform()
print("Number of Selected Features: ", len(selected))
Xc  = sm.add_constant(encoded_df[selected])
model   = sm.OLS(y, Xc)
results = model.fit()            
print(results.summary())
logl  = model.loglike(results.params)
model_df = model.df_model + 2 #plus intercept and sigma
nobs     = y.shape[0]
bic      = em.bic(logl, nobs, model_df)
aic      = em.aic(logl, nobs, model_df)
print("BIC: ", bic)
print(" ")    
print("{:*>71s}".format('*'))
print("{:*>14s}     REPLACE IMPUTE ENCODE      {:*>25s}". format('*', '*'))
print("{:*>71s}".format('*'))

rie = ReplaceImputeEncode(data_map=attribute_map, 
                          nominal_encoding="one-hot", 
                          drop=True, display=True)
encoded_df = rie.fit_transform(df)

y = encoded_df[target]
X = encoded_df.drop(target, axis=1)
print(" ")
print("{:*>71s}".format('*'))
print("{:*>14s}     FIT FULL MODEL        {:*>30s}". format('*', '*'))
print("{:*>71s}".format('*'))
Xc = sm.add_constant(X)
lr = sm.OLS(y, Xc)
results = lr.fit()
print(results.summary())
ll       = lr.loglike(results.params)
model_df = lr.df_model + 2 #plus intercept and sigma
nobs     = y.shape[0]
bic      = em.bic(ll, nobs, model_df)
print("BIC:", bic)
print(" ")
print("{:*>71s}".format('*'))
print("{:*>18s}        LASSO       {:*>33s}". format('*', '*'))
print("{:*>71s}".format('*'))
alpha_list = [0.00001, 0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 
              0.006, 0.007, 0.008, 0.009, 0.01, 0.1, 1.0]
for a in alpha_list:
    clr = Lasso(alpha=a, random_state=12345)
    clr.fit(X, y)
    c = clr.coef_
    z = 0
    for i in range(len(c)):
        if abs(c[i]) > 1e-3:
            z = z+1
    print("\nAlpha: ", a, " Number of Coefficients: ", z, "/", len(c))
    linreg.display_metrics(clr, X, y)

print("{:*>71s}".format('*'))
