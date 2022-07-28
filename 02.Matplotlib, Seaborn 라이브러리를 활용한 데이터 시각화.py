#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
# https://www.inflearn.com/course/%EB%82%98%EB%8F%84%EC%BD%94%EB%94%A9-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D-%EC%8B%9C%EA%B0%81%ED%99%94
from timeit import timeit
from turtle import color
from black import syms
from matplotlib import legend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
# matplotlib 파이썬으로 기본적인 차트들을 쉽게 그릴 수 있도록 도와주는 시각화 라이브러리
# seaborn : matplotlib 기반으로 만들어진 통계 데이터 시각화 라이브러리

import warnings
sns.set(style='whitegrid')
sns.set_palette('pastel')
warnings.filterwarnings('ignore')

# %%
tips = sns.load_dataset('tips')
tips

# %%
sum_tip_by_day = tips.groupby('day')['tip'].sum()
x_label = ['Thu','Fri','Sat','Sun']

x_label_index = np.arange(len(x_label))
x_label_index

# %%
plt.bar(x_label, sum_tip_by_day,
        color='pink', # color : 색깔 지정
        alpha=0.6, # alpha : 색 투명도
        width=0.3, # width:0.3
        align='center') # default:'center'
plt.title('Sum Of Tips by Days', fontsize=16)
plt.xlabel('Days',fontsize=14)
plt.ylabel('Sum Of Tips',fontsize=14)
plt.xticks(x_label_index,
           x_label,
           rotation=45, # x 라벨이 많은 경우 기울여서 씀
           fontsize=15)
plt.show()

# %%
# seaborn을 활용한 시각화
# sns.barplot(data = tips, # 데이터 프레임
#             x='day', # x 변수
#             y='tip', # y 변수ns.barplot(data = tips, # 데이터 프레임
#             hue='sex', # 특정 컬럼값을 기준으로 나눠서 보고 싶을 때
#             palette='pastel', # pastel, husl, Set2, flare, Blues_d
#             order=['Sun','Sat','Fri','Thur'], # x 변수 순서 지정
#             edgecolor=".6", # edge 선명도 지정
#             linewidth=2.5 # line두께 지정
#             )
sns.barplot(data = tips,
            x='day',
            y='tip',
            estimator=np.sum,
            hue='smoker' # 비교 컬럼 항목
)
plt.title('Sum Of Tips by Days', fontsize=16)
plt.xlabel('Days')
plt.ylabel('Sum of Tips')
plt.xticks(rotation=45)
plt.show()

# %%
# 파이챠트를 활용한 시각화
# matplotlib을 활용한 시각화
sum_tip_by_day = tips.groupby('day')['tip'].sum()
sum_tip_by_day

# %%
ratio_tip_by_day = sum_tip_by_day/sum_tip_by_day.sum()
ratio_tip_by_day

# %%
x_label = ['Thu','Fri','Sat','Sun']
plt.pie(ratio_tip_by_day, # 비율 값
        labels=x_label, # 라벨 값
        autopct='%.1f%%', # 부채꼴 안에 표시될 숫자 형식(소수점 1자리까지 표시)
        startangle=90, # 축이 시작되는 각도 설정
        counterclock=True, # True: 시계방향순 , False:반시계방향순
        explode=[0.05, 0.25, 0.05, 0.05], # 중심에서 벗어나는 정도 표시
        shadow=True, # 그림자 표시 여부
        colors = ['#ff9999', '#ffc000', '#8fd9b6', '#d395d0'], # colors=['gold','silver','whitesmoke','gray']
        wedgeprops = {'width':0.7,'edgecolor':'w','linewidth':3}
        ) #width: 부채꼴 영역 너비,edgecolor: 테두리 색 , linewidth : 라인 두께
plt.title('Ratio Of Tips by Days', fontsize=16)
plt.show()

# %%
# 라인 챠트
# line 차트 예제를 위해, tips 데이터에 가상 시간 컬럼 추가하기
# 일요일 데이터만 사용
sun_tips = tips[tips['day']=='Sun']
sun_tips

# %%
# 현재 서버 시간을 얻기 위해 datetime 라이브러리 사용
import datetime

today = datetime.date.today() # 오늘 날짜 출력 YYYY-MM-DD
today

# %%
date = []
date.append(today)
date

# %%
for i in range(sun_tips.shape[0]-1):
    today += datetime.timedelta(+1) # 하루씩 추가
    date.append(today)
sun_tips['date'] = date

plt.plot(sun_tips['date'],sun_tips['total_bill'],
         linestyle='-', # linestyle= '--', '-', ':', '-.
         linewidth=1, # line두께
         color='pink', # 색상 선택
         alpha=0.9, # 투명도 조절
         )
plt.title('Total Tips by Date',fontsize=20)
plt.xlabel('date',fontsize=15)
plt.ylabel('total tip',fontsize=15)
plt.xticks(rotation=90)
plt.show()


# %%
# seaborn을 활용한 시각화
sns.lineplot(data=sun_tips,x='date',y='total_bill',
             hue='sex',
             palette='pastel')
plt.title('Total Bill by Date & Sex')
plt.show()


# %%
# Scatter 및 HeatMap 챠트 이해
# Scatter 챠트 -> 상관 관계 분석 (기온에 따른 아이스크림 판매)

# matplotlib을 활용한 시각화
plt.scatter(tips['total_bill'], tips['tip'],
            color='pink', # 색상 선택
            edgecolor='black', # 테두리 색깔
            linewidth=2) # 라인 두께
plt.show()

# %%
# seaborn을 활용한 시각화
sns.scatterplot(data=tips,
                x='total_bill',
                y='tip',
                hue='day',    # 색깔로 구분해서 보고 싶을 때
                style='time', # 모양으로 구분해서 보고 싶을 때
                size='size',  # 크기로 구분해서 보고 싶을 때
                sizes=(20, 400), # 크기별로 보고 싶을때
                palette='pastel') # 색상 지정
plt.title('Scatter between total_bill and tip',fontsize=20)
plt.xlabel('total_bill',fontsize=16)
plt.ylabel('tip',fontsize=16)
plt.legend(loc=(1.01, 0.495))
plt.show()

# %%
# seaborn을 활용한 시각화
# https://seaborn.pydata.org/generated/seaborn.heatmap.html
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
sns.heatmap(tips.corr(),
            annot=True,  # 박스안 상관관계값 표시할지 말지
            square=True, # 박스를 정사각형으로 할지 말지
            vmin=0.3,vmax=1, # 최소 최댓값 지정
            linewidth=0.5, # 라인 두께 지정
            cmap='RdYlBu' # 색상 선택 (RdBu, RdGy, RdYlBu, RdYlGn, BrBG, PuOr, PuBuGn, PuBu, PuRd, PuBuRd, Purples, Blues, Greens, Greys, YlGn, YlGnBu, YlOrBr, YlOrRd, afmhot, bone, cool, copper, flag, gist_heat, gist_gray, gist_yarg, gray, hot, jet, pink, prism, rainbow, seismic, spectral, spring, summer, winter, ylgn, ylgnbu, ylorbr, ylorrd)
            )
plt.title('Heatmap by correlation',fontsize=20)
plt.show()

# %%
# 히스토그램 : 변수에 대한 분포를 알아볼 경우 사용
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
# matplotlib을 활용한 시각화
plt.hist(tips['total_bill'],
         bins=30, # 잘게 분해해서 보고 싶을 때
         density=True, # 빈도수로 표현 (비율로 표현)
         alpha=0.7,
         color='pink',
         edgecolor='black',
         linewidth=0.9)
plt.title('Histogram for total_bill')
plt.xlabel('total_bill')
plt.ylabel('rate')
plt.show()

# %%
# seaborn을 활용한 시각화
sns.histplot(data=tips,
             x='total_bill', # y 축은 별도 정의가 필요 없습니다.
             bins=30, # bin의 갯수
             kde=True, # 라인으로 분포표 표시 여부
             hue='sex', # 색깔에 따른 구분
             multiple='stack', # dodge, stack
             stat="density", # 비율로 확인할 때
             shrink=0.6 # bin의 두께
             )
plt.title('Histogram for total_bill')
plt.xlabel('total_bill')
plt.ylabel('rate')
plt.show()


# %%
# 박스 챠트
# matplotlib을 활용한 시각화
plt.boxplot(tips['tip'],
            sym='rs', # outlier(이상치) => red & square로 표현
          )
plt.title('Box Plot for Tip',fontsize=20)
plt.ylabel('tip',fontsize=15)
plt.show()

# %%
# seaborn을 활용한 시각화
# flierprops : 이상치 속성
flierprops = dict(marker='o', markerfacecolor='r', markersize=10,  markeredgecolor='black')
sns.boxplot(data=tips,
            x='day',
            y='tip',
            hue='smoker',
            palette='pastel',
            linewidth=1,
            flierprops=flierprops,
            order=["Sun", "Sat","Fri","Thur"])
plt.title('Box Plot for Tip by Day',fontsize=20)
plt.ylabel('tip',fontsize=15)
plt.show()

# 단순히 보면 보이지 않던 것들이 비로소 시각화를 하면 새로운 인사이트를 발견하게 된다.