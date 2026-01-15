import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import shap
from joblib import load
import matplotlib
from sklearn.preprocessing import StandardScaler
import sklearn
import os

#打印出上述包的版本号
print("streamlit version:", st.__version__)
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("matplotlib version:", matplotlib.__version__)
print("catboost version:", CatBoostClassifier.__version__)
print("shap version:", shap.__version__)
print("joblib version:", load.__version__)
print("sklearn version:", StandardScaler.__version__)
