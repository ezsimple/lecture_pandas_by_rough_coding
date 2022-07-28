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
# 간단히 메트릭스를 생성하는 방법
matrics = np.arange(12).reshape(3,4)
matrics

# %%
# 메트릭스 정보를 이용해서 DataFrame 생성 및 컬럼 삭제
df = pd.DataFrame(matrics, columns=['A','B','C','D'])
df = df.drop(['A', 'B'], axis=1) # axis=0:index(default), axis=1:column
df

# %%
# csv 파일을 읽어서 DataFrame 생성
df = pd.read_csv('data/friend.csv')
df

# %%
# index 2 번에 해당하는 row 가져 오기
df.iloc[2]

# column job에 해당하는 데이터 가져오기
df['job']

# %%
df.loc[:,'job']

# %%
# 슬라이싱 기능을 통해 여러 행 가져오기
df.loc[0:2]

# %%
# loc와 iloc의 차이를 꼭 기억하세요.
df.iloc[0:2]

# %%
# 조건 필터링 가져오기
o = df[df['age'] > 30]

# %%
# job이 인턴인 사람을 가져오기
o = df[df['job'] == 'intern']

# %%
# 30대 이상, 40대 이하 인원을 가져오기
# 괄호가 없으면 에러남 (& 연산자 우선순위에 의해 애매함)
df[(df['age'] >= 30) & (df['age'] <= 40)]

# 30대 미만, 40대 초과 인원을 가져오기
df[(df['age'] < 30) | (df['age'] > 40)]

# %%
4 in [1,2,3,5] # left side is element, right side is list

# %%
# 중요 : in operation by lambda: left side is list, right side is element
df[df['job'].apply(lambda x: x in ['intern', 'student'])]

# %%
metrics = np.arange(12).reshape(3,4)
df = pd.DataFrame(metrics, columns=['A','B','C','D'])
df

# %%
# B, C 컬럼을 삭제합니다.
df.drop(['B', 'C'], axis=1)

# index 0, 1을 삭제합니다. axis=0은 default
df.drop([0, 1])

# %%
# 행, 열 수정
df = pd.read_csv('data/friend.csv')
df

# %%
# 복제하기
temp = df.copy()
temp

# %%
temp['age'] = 20
temp.loc[2, 'age'] = 30
temp


# %%
df = pd.read_csv('data/abalone.data',
             header=None,sep=',',
             names=['sex','length','diameter','height',
                  'whole_weight','shucked_weight','viscera_weight',
                  'shell_weight','rings'])

# 1. 데이터 형태를 확인
# 딥러닝(CNN) => 이미지(데이터수, 가로, 세로, 채널)
# 딥러닝(RNN) => 텍스트, 시계열 (데이터수, 시간, 세로)
df.shape

# %%
# 2. 결측치 확인(중요)
# 빈값을 꼭 확인 해야 합니다. sum()을 이용해서 결측치 확인
df.isnull().sum().sum()

# %%
# Series의 sum을 이용해서, 결측치 확인
pd.Series([True, False]).sum()

# %%
# 기술통계확인(평균, 표준편차, 크기비율, 최빈값, 중앙값, 분산)
# numeric은 숫자형의 데이터만 가져 옵니다.
df.describe()

# %%
gdf = df['whole_weight'].groupby(df['sex'])
gdf.describe()

# %%
gdf.sum()

# %%
gdf.size()

# %%
# 그룹변수가 하나가 아닌, 다른 변수에 대한 집계
df.groupby(df['sex']).mean()

# %%
df.groupby(['sex']).mean()

# %%
# np.where 를 이용해 '3중 조건문'을 만들 수 있습니다.
df['length_bool'] = np.where(df['length']>df['length'].median(),
                            '큰전복', # True일 경우
                            '작은전복' # False일 경우)
                            )
df['length_bool']

# %%
# 순서대로 groupby, EDA(탐색적 분석 과정)을 수행합니다.
df.groupby(['sex','length_bool']).size()

# %%
# 중복된 row를 확인 하는 것 (현업에서는 거의 없는 경우)
df.duplicated().sum()

# %%
new_df = df.iloc[[0]] # 새로운 DataFrame
dup_df = pd.concat([df, new_df]) # 중복된 row를 추가

dup_df.duplicated().sum()

# %%
dup_df.drop_duplicates(inplace=True) # 중복된 row를 삭제
dup_df.duplicated().sum()

# %%
# 결측치를 찾아서 다른 값으로 변경
nan_df = dup_df.copy()
nan_df.loc[2, 'length'] = np.nan
nan_df.loc[4, 'sex'] = np.nan
nan_df.isnull().sum()

# %%
# fillna()를 이용해서 결측치를 채워줍니다. (중요)
nan_df.fillna(nan_df.mean(), inplace=True)
nan_df.isnull().sum()

# %%
# sex와 같은 카테고리형 변수의 경우, 결측치를 채울 mode 값을 구하는 방법
sex_mode = nan_df['sex'].value_counts().sort_values(ascending=False).index[0]
sex_mode
nan_df['sex'].fillna(sex_mode, inplace=True)
nan_df.isnull().sum()

# %%
# apply()를 이용해서 결측치를 채워줍니다. (중요!!!)
# DataFrame타입의 객체에서 호출가능한 apply함수에 대해 살펴보자.
# 본인이 원하는 행과 열에 연산 혹은 function을 적용할 수 있다.

# 열 기준으로 집계하고 싶은 경우 axis=0
# 행 기준으로 집계하고 싶은 경우 axis=1
adf = df.copy()
adf['diameter']


# %%
# 사용자 함수를 통한 집계

import math

def avg_ceil(x, y, z):
  return math.ceil((x+y+z)/3)

# %%
# lambda를 통한 집계 (중요 !!!!)
adf[['diameter', 'height', 'length']].apply(lambda x: avg_ceil(x[0],x[1],x[2]), axis=0)

# %%
# 문제: 세 변수의 합이 1을 넘으면 True, 아니면 False 출력후 answer 변수로 저장
# adf에 answer 컬럼을 추가하고 입력

# sum(list), not list.sum().
def answer(x):
  return sum(x) > 1

answer([0, 1, 2])
adf['answer'] = adf[['diameter', 'height', 'length']].apply(lambda x: answer([x[0],x[1],x[2]]), axis=1)
adf


