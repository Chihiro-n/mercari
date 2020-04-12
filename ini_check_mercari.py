import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling as pdp
import seaborn as sns

# tsvファイルの読み込み csvではないので、delimiterオプションを設定
df_train = pd.read_csv('/Users/apple/python/kaggle/mercari/mercari-price-suggestion-challenge/train.tsv',delimiter='\t' ,index_col=0)

print(df_train.head())

print(df_train.columns)
print(df_train.shape)

# df_trainのカgit fetch originテゴリ名、商品説明、投稿タイトル、ブランド名のデータタイプを「category」へ変換する
df_train.category_name = df_train.category_name.astype('category')
df_train.item_description = df_train.item_description.astype('category')
df_train.name = df_train.name.astype('category')
df_train.brand_name = df_train.brand_name.astype('category')

# タイプがどうなっているか確認
print(df_train.dtypes)

print("\n")
print("\n")
print("\n")
print("\n")

### ユニークであるのかの確認 かぶりがないかの確認
print( df_train.apply(lambda x: x.nunique()) )

print("\n")
print("\n")
print("\n")
print("\n")

#欠損値の有無
print( df_train.isnull().any(axis=0) )

pd.options.display.float_format = '{:.2f}'.format

#基本統計量のチェック
print( df_train.describe() )



