# Sage Math Cloundを使った演習

[Sage Math Cloud](http://cloud.sagemath.com)を使ってPythonを動かす実習を行う。
以下の例のいくつかは「データサイエンティスト機械学習入門編」（以下「養成読本」と略す）掲載のソースコードを一部手直ししたもので、それらの動作の仕組みの詳細については養成読本を参照のこと。

##　乱数データを使った線形回帰の実験
（養成読本参照）

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import linear_model, datasets

# 乱数によりデータを生成
np.random.seed(0)
regdata = datasets.make_regression(100, 1, noise=20.0)

# 学習を行いモデルのパラメータを表示
lin = linear_model.LinearRegression()
lin.fit(regdata[0], regdata[1])
print("coef and intercept :", lin.coef_, lin.intercept_)
print("score :", lin.score(regdata[0], regdata[1]))

# グラフを描画
xr = [-2.5, 2.5]
plt.plot(xr, lin.coef_ * xr + lin.intercept_)
plt.scatter(regdata[0], regdata[1])
```

## 糖尿病データを使った予測
（養成読本参照）

```python
from sklearn import linear_model, datasets

# データの読み込み
diabetes = datasets.load_diabetes()

# データを訓練用と評価用に分ける
data_train = diabetes.data[:-20]
target_train = diabetes.target[:-20]
data_test = diabetes.data[-20:]
target_test = diabetes.target[-20:]

# 学習させる
lin = linear_model.LinearRegression()
lin.fit(data_train, target_train)

# 当てはまり度合いを表示
print("Score :", lin.score(data_test, target_test))

# 最初の評価用データについて結果を予想して、実際の値と並べて表示
print("Prediction :", lin.predict(data_test[0].reshape(1,-1)))  # 予想
print("Actual value :", target_test[0])  # 実際の値
```

## あやめデータによる二値分類

```python
import sklearn.datasets as datasets
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

# データの読み込み
iris = datasets.load_iris()

# 種類が2であるものを捨てる
data = iris.data[iris.target != 2]
target = iris.target[iris.target != 2]

# ロジスティック回帰による学習と交差検定による評価
logi = LogisticRegression()
scores = cross_validation.cross_val_score(logi, data, target, cv=5)

# 結果を表示する
print(scores)
```
