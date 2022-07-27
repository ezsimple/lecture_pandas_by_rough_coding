#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
# https://www.inflearn.com/course/%EB%82%98%EB%8F%84%EC%BD%94%EB%94%A9-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D-%EC%8B%9C%EA%B0%81%ED%99%94
from timeit import timeit
from turtle import color
from matplotlib import legend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import matplotlib as mpl

# 한글화 작업
plt.figure(dpi=600) # 그래프를 선명하게
plt.rc('font', family = 'NanumGothic') # 시스템에 폰트설치후, 시스템 재시작
plt.rc('axes', unicode_minus = False) # 한글 폰트 사용시 마이너스 폰트가 깨지는 문제 해결
plt.style.use('fivethirtyeight') # 스타일을 사용해 봅니다.

# 판다스 최대 컬럼 지정 (컬럼에 ... 표시 방지)
pd.options.display.max_columns = 100
# retina 디스플레이가 지원되는 환경에서 시각화 폰트가 좀 더 선명해 보입니다.
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('retina')

# %%