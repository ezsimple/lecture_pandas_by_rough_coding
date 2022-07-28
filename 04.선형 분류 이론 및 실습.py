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
# plt.style.use('fivethirtyeight') # 스타일을 사용해 봅니다.

# 판다스 최대 컬럼 지정 (컬럼에 ... 표시 방지)
pd.options.display.max_columns = 100
# retina 디스플레이가 지원되는 환경에서 시각화 폰트가 좀 더 선명해 보입니다.
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('retina')

# %%
# - 로지스틱 회귀란 무엇인가

# 로지스틱 회귀

# 로지스틱 회귀란 샘플이 특정 클래스에 속할 확률을 추정하는 것,
# ex)특정 이메일이 스팸일 확률

# binary한 문제일 경우, 추정 확률이 50%가 넘으면 모델은
# 그 샘플이 해당 클래스에 속한다고 할 수 있음(이진 분류기)
# 확률을 추정하는 법

# 선형 회귀 모델과 같이 로지스틱 회귀 모델도 마찬가지로 입력 변수의 가중치 합을 계산한다.
# 대신 선형회귀와 같이 결과물을 연속적인 형태로 출력하는 것이 아니라, 0~1사이의 확률 값을 출력한다.
# 로지스틱 함수

# - f(x)값의 산출물인 p(probability)< 0.5 인 경우, y = 0
# - f(x)값의 산출물인 p(probability)>= 0.5 인 경우, y = 1

# - 로지스틱 회귀 모델의 훈련과 비용함수
# 로지스틱 회귀 모델의 훈련 목적은
# positive(y=1)에 대한 샘플에 대해서는 높은 확률로 추정토록 하고,
# negative(y=0)에 대한 샘플에 대해서는 낮은 확률로 추정하게 하는 모델 최적의 가중치를 찾는 것입니다.


# %%
# 목표 : regression과 classification 이론 및 모형들 통해서
# 전반적인 supervised learning의 이해 및 인지적 구조가 형성

# 로지스틱 회귀를 더욱 이해하기 쉽게 설명하기 위해서 붗꽃 데이터 사용
# 이 데이터는 세 개의 품종 (Setosa, Versicolor, Virginica), 150개의 데이터 수, Petal(꽃잎) Sepal(꽃받침)의 너비와 길이를 가진다.

from sklearn import datasets
iris = datasets.load_iris() # sklearn의 빌트인 iris 예제 로드
print(list(iris.keys())) # iris데이터 key 값

# %%
x = iris['data'][:,3:] # 꽃잎의 너비 변수만 사용하겠다.
y = (iris['target']==2).astype('int') # index=2 : Versinica

# %%
# 로지스틱 회귀 : 선형 회귀와 동일하게 입력 변수의 가중치 합을 계산한다.
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x,y)

# %%
np.linspace(0,3,1000).reshape(-1,1)

# %%
x_new = np.linspace(0,3,1000).reshape(-1,1)
# predict VS predict_proba => predict: 예측 라벨값 산출, predict_proba: 예측 확률 값 산출
y_proba = log_reg.predict_proba(x_new)
plt.plot(x_new, y_proba[:,1], "r-", label = "Iris-Virginica")
plt.plot(x_new, y_proba[:,0], "b--", label = "not Iris-Virginica")
plt.legend()
plt.show()


# %%
# 소프트맥스 회귀 : 로지스틱 회귀와 비슷하지만, 예측 확률을 사용하지 않는다.

x = iris['data'][:,(2,3)] # 꽃잎의 길이, 너비 변수 사용
y = iris['target'] # 3개 클래스 모두 사용

# multi class 역시 sklearn의 logisticregression 사용
# multi_class = 'multinomial' 옵션으로 소프트맥트 회귀를 사용할 수 있음
# solver = 'lbfgs'의 lbfgs는 의사 뉴턴 메서드 중, 제한된 메모리 공간에서 구현한 것으로 머신러닝 분야에서 많이 사용 됨
# 하이퍼파라미터 C를 통해, 이전 장에서 배운 L2 규제를 사용하게 됨
softmax_reg = LogisticRegression(multi_class='multinomial',solver='lbfgs',C=10,random_state=2021)
softmax_reg.fit(x,y)

# 꽃잎 길이 5cm, 너비 2cm의 iris 데이터를 예측한다고 가정
new_iris = [[5,2]]
prediction = softmax_reg.predict(new_iris)[0]
label = iris['target_names'].tolist()
print(label[prediction])

softmax_reg.predict_proba(new_iris)


# %%
# 서포트 벡터 머신(SVM) : (인기있는 머신러닝 모델)
# SVM은 특성 스케일에 아주 민감하다. (sklearn.StandardScaler를 사용하여 특성 스케일을 정규화해야 한다.)
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris() # 사이킷런 빌트인 iris 예제 데이터 로드
x = iris['data'][:,(2,3)] # 꽃잎의 길이, 너비 변수 사용
y = (iris['target']==2).astype('int') # index=2 : Versinica

# 사이킷런의 파이프라인 라이브러리를 통해서 데이터 스케일과 모델 적합을 한번에 할 수 있음.
svm_clf = Pipeline([
                    ('scaler',StandardScaler()),
                    ('linear_svc',LinearSVC(C=1,loss='hinge'))
])

# 모델 훈련
svm_clf.fit(x,y)

# 꽃잎 길이 5.5cm, 너비 1.7cm의 iris 데이터를 예측한다고 가정
new_iris = [[5.5,1.7]]
prediction = svm_clf.predict(new_iris)[0]
print(prediction) # 1 : Versinica : True(1)

# 비선형 SVM 분류
# 선형 SVM이 많은 경우에서 잘 작동하지만, 데이터셋 자체가 선형으로 잘 분류할 수 없는 경우도 많다.
# 간단히 이러한 데이터셋에서는 다항 특성(polynomial)인 아래와 같은 특성을 추가하면 된다.
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rc('axes', unicode_minus = False) # 한글 폰트 사용시 마이너스 폰트가 깨지는 문제 해결

x = np.linspace(-3,3,10)
y = np.linspace(1,1,10)

# %%
plt.scatter(x,y)
plt.grid()
plt.show()

# %%
# x^2의 basis function을 통해 구현
b_func = x**2
plt.scatter(x,b_func)
plt.grid()
plt.show()


# %%
# hinge loss

from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC

# 샘플 수 1000개, noise값 0.1, random_state=2021
x,y = make_moons(n_samples=1000,noise=0.1,random_state=2021)

# 변수변환 : polynomial 3차 다항식 사용, scaler : StandardScaler 사용, 모델 : LinearSVM(C=10,loss='hinge) 사용
polynomial_std_svm = Pipeline([
                               ("polynomial",PolynomialFeatures(degree=3)),
                               ("std",StandardScaler()),
                               ("svm",LinearSVC(C=10,loss='hinge'))
])

# %%
# 모델 학습
polynomial_std_svm.fit(x,y)

# %%
# 첫번째 변수값:2.0, 두번째 변수값:1.0 인 새로운 데이터 예측
new_moon = [[2.0,1.0]]
polynomial_std_svm.predict(new_moon)


# %%
# - 커널(다항식, 가우시안 RBF)
# ✔ 다항식 커널
from sklearn.svm import SVC
# kernel='poly(degree=3)'사용
# 매개변수 coef0는 모델이 높은 차수와 낮은 차수에 얼마나 영향을 받을지 조절하는 것
# coef0을 적절한 값으로 지정하면 고차항의 영향을 줄일 수 있다. (coef0의 default=0)
poly_kernel_std_svm = Pipeline([
                            ("std",StandardScaler()),
                            ("poly_kernel_svm",SVC(kernel='poly',degree=3,coef0=1,C=5))
])
poly_kernel_std_svm.fit(x,y)


# %%
# ✔ 가우시안 RBF 커널
# 하이퍼파라미터 r는 규제 역할을 한다.
# (모델이 과적합일 경우=> r 감소시키고, 모델 과소적합일 경우=> r 증가시켜야함)
# 하이퍼파라미터 C도 r(gamma)와 비슷한 성격을 띈다.
# 그래서 모델 복잡도를 조절하기 위해서 gamma와 C를 함께 조절해야 한다.
# Tip (하이퍼파라미터 조절) : 그리드 탐색법 사용(그리드 큰 폭 => 그리드 작은 폭) : 줄여가면서 탐색
rbf_kernel_std_svm = Pipeline([
                               ('std',StandardScaler()),
                               ('rbf_kernel_svm',SVC(kernel='rbf',gamma=3,C=0.001))
])
rbf_kernel_std_svm.fit(x,y)