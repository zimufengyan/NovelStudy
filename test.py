# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      test
   Author :         zmfy
   DateTime :       2023/12/16 19:14
   Description :    
-------------------------------------------------
"""
import pandas as pd

df = pd.read_parquet('./data/test.parquet')
print(df.head())