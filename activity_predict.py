# -*- coding: utf-8 -*-

import pandas as pd
import os

# init data

xx = 20
print('xx', xx)

datadir = "data/UCI_HAR_Dataset/"

feature_file = datadir + "features.txt"
dfcol = pd.read_csv(feature_file, sep='\s+', header=None, index_col=0)
print('dfcol\n', dfcol)


