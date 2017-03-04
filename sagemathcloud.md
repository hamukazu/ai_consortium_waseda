# Sage Math Cloundを使った演習

[Sage Math Cloud](http://cloud.sagemath.com)を使ってPythonを動かす実習を行う。

## <a name="a">プロジェクトとPython実行用ノートブックの作成</a>

まずプロジェクトを作る

1. 「Create new project」ボタンを押す
1. 「Title」に適当な名前を入力、「Create project without upgrades」を押す

次にノートブックを作る

1. 「Create」ボタンを押す
1. （一番上にタイトルを入力してもいいが、そのままデフォルトでもいい））
1. 「Jupyter Notebook」ボタンを押す


## <a name="b">簡単な例</a>
以下の例は「データサイエンティスト機械学習入門編」（以下「養成読本」と略す）掲載のソースコードを一部手直ししたもので、それらの動作の仕組みの詳細については養成読本を参照のこと。

###　乱数データを使った線形回帰の実験

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
plt.rcParams["figure.figsize"] = (14,10)


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

### 糖尿病データを使った予測

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

### あやめデータによる二値分類

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

### 交差検定の例

```python
from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation

# データの読み込み
iris = datasets.load_iris()

# 学習
svc = svm.SVC()
scores = cross_validation.cross_val_score(svc, iris.data, iris.target, cv=5)

# 結果表示
print(scores)
print("Accuracy:", scores.mean())
```

### あやめの分類の可視化

```python
from sklearn import datasets
from sklearn import svm
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (14,10)

# データの読み込み
iris = datasets.load_iris()

# PCAによるデータ変換
pca = PCA(n_components=2)
data = pca.fit(iris.data).transform(iris.data)

# メッシュ作成
datamax = data.max(axis=0) + 1
datamin = data.min(axis=0) - 1
n = 200
X, Y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
                   np.linspace(datamin[1], datamax[1], n))

# 分類
svc = svm.SVC()
svc.fit(data, iris.target)
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])

# 描画
plt.contourf(
    X, Y, Z.reshape(X.shape), levels=[-0.5, 0.5, 1.5, 2.5],
    colors=["r", "g", "b"])
for i, c in zip([0, 1, 2], ["r", "g", "b"]):
    d = data[iris.target == i]
    plt.scatter(d[:, 0], d[:, 1], c=c)
```

## <a name="c">簡易レコメンデーションシステムの実装</a>

[こちら](https://hamukazu.github.io/ai_consortium_waseda/recommender.html)を参照。

（さらに大規模データを使った例は[こちら](https://hamukazu.github.io/ai_consortium_waseda/recommender_big_data.html)参照）

## <a name="d">日経平均株価の予測</a>

[こちら](https://hamukazu.github.io/ai_consortium_waseda/nikkei_index.html)を参照。

