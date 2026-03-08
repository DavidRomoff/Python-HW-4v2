
import pytest
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import sklearn

import MyLM_File
def test_MyLM():
    data = sklearn.datasets.load_iris()
    iris = pd.DataFrame(data=data.data, columns=data.feature_names)
    iris.columns = iris.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    myFormula = "sepal_length_cm ~ petal_length_cm"
    x = MyLM(myFormula, iris)
    assert np.round(x.params.iloc[0], 4) == 4.3066

import TMAT1_File
def test_TMAT1():
    rLast = np.repeat(['A', 'B', 'C'], [3, 4, 5])
    rNow = np.repeat(['A', 'B', 'C'], [5, 2, 5])
    out = TMAT1(rLast, rNow)
    answer = np.array([1, 0, 0, 0.5, 0.5, 0, 0, 0, 1]).reshape(3, 3)
    row_sums = out.sum(axis=1)
    out = out.div(row_sums, axis=0)
    assert np.array_equal(out, answer)

import Forecast_nPeriod_File
def test_Forecast_nPeriod():
    initialStates = np.array([[20, 30, 10]])
    tmat = np.array([[1, 0, 0], [.5, .5, 0], [0, 0, 1]]) 
    states1 = Forecast_nPeriod(initialStates, tmat, 1)
    states2 = Forecast_nPeriod(initialStates, tmat, 2)
    states3 = Forecast_nPeriod(initialStates, tmat, 3)
    assert np.isclose(states1[0][0], 35)
    assert np.isclose(states2[0][0], 42.5)
    assert np.isclose(states3[0][0], 46.25)
