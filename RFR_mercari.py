import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling as pdp
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
pd.set_option('display.float_format', lambda x:'%.5f' % x)

# tsvファイルの読み込み csvではないので、delimiterオプションを設定
### 訓練データにはpriceがあり、テスト用データにはpriceのカラムがない
df_train_law = pd.read_csv('/Users/apple/python/kaggle/mercari/mercari-price-suggestion-challenge/train.tsv',delimiter='\t' ,index_col=0)
df_test_law = pd.read_csv('/Users/apple/python/kaggle/mercari/mercari-price-suggestion-challenge/test.tsv',delimiter='\t' ,index_col=0)

######################################################################
####  訓練データとテストデータを一つのDFにして RandomForestRegression    ###
######################################################################

### 訓練用と、テスト用でカラム名が異なるので統一させる
df_train_law = df_train_law.rename(columns = {'train_id':'id'})
df_test_law = df_test_law.rename(columns = {'test_id':'id'})
 
### 訓練用か否かを判別するカラムを新しく作る
df_train_law['is_train'] = 1
df_test_law['is_train'] = 0
 
### 訓練用からpriceカラムを消去
df_droped_price = df_train_law.drop(['price'], axis=1)

### 訓練用とテスト用を連結
df_combine = pd.concat([df_droped_price, df_test_law],axis=0)

### train_test_combineの文字列のデータタイプを「category」へ変換
df_combine.category_name = df_combine.category_name.astype('category')
df_combine.item_description = df_combine.item_description.astype('category')
df_combine.name = df_combine.name.astype('category')
df_combine.brand_name = df_combine.brand_name.astype('category')
 
# combinedDataの文字列を「.cat.codes」で数値へ変換する
df_combine.name = df_combine.name.cat.codes
df_combine.category_name = df_combine.category_name.cat.codes
df_combine.brand_name = df_combine.brand_name.cat.codes
df_combine.item_description = df_combine.item_description.cat.codes
 
print( df_combine.head() )

### Ramdom Forest Regressionをする前の準備として trainとtestに分ける

df_test = df_combine.loc[df_combine['is_train'] == 0]
df_train = df_combine.loc[df_combine['is_train'] == 1]
 
# 「is_train」をtrainとtestのデータフレームから落とす
df_test = df_test.drop(['is_train'], axis=1)
df_train = df_train.drop(['is_train'], axis=1)


# df_trainへprice（価格）を戻す
df_train['price'] = df_train_law.price
print( "price チェック" )
print( df_train.head() )

# price（価格）をlog関数で処理
### 対数変換を行う前のスコア 0.6292
### 対数変換を行ったあとのスコア 0.7223
### だから対数変換は行おう
df_train['price'] = df_train['price'].apply(lambda x: np.log(x) if x>0 else x)
print( "log 変換後 price チェック" )
print( df_train.head() )


# x ＝ price以外の全ての値、y = price（ターゲット）で切り分ける
x_train = df_train.drop(['price'], axis=1)
y_train = df_train.price

print(x_train.shape)
print(y_train.shape)

# モデルの作成
m = RandomForestRegressor(n_jobs=-1, min_samples_leaf=5, n_estimators=200)
m.fit(x_train, y_train)
# スコアを表示
print( m.score(x_train, y_train) )

# 作成したランダムフォレストのモデル「m」に「df_test」を入れて予測する
### 対数の予測価格
preds = m.predict(df_test)
print(preds)

### 対数を指数変換して、もとの価格に戻す
preds = pd.Series(np.exp(preds))
print(preds)

### 変数の影響度を求める
feature_importances = m.feature_importances_
print(feature_importances)

plt.figure(figsize=(10, 5))
plt.ylim([0, 1])
y = feature_importances
x = np.arange(len(y))
plt.bar(x, y, align="center")
plt.xticks(x, x_train.columns.values)
plt.show()