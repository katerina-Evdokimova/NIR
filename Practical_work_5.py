# Commented out IPython magic to ensure Python compatibility.
#импортируем внешние модули и библиотеки
import warnings
warnings.filterwarnings("ignore")

import pandas as pd # библиотека для работы с наборами данных
import matplotlib.pyplot as plt # библиотека для визуализации
import numpy # структура данных ndarray, статистики (хотя там много всего)

import seaborn as sns # более продвинутая библиотека для визуализации данных
sns.set(style="white", color_codes=True)

# чтобы изображения отображались прямо в ноутбуке
# %matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

#считаем данные и посмотрим на первые 5 строк
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/2015-street-tree-census-tree-data.csv")
data.head()

#пропуски в данных
data.info() #крайне удивительно, но пропусков у нас нет. 
# Практически идеальный датасет

#избавимся от ненужных столбцов
data = data.drop('state', 1) # данный столбец нам не интересенб т.к. он везде одинаковый и имеющий значение New York
data = data.drop('user_type', 1) # данный столбец нам не интересень т.к. не важно кто собрал информацию
data = data.drop('created_at', 1) # нам не важна дата сбора информации о дереве

#общая статистика по каждому столбцу
data.describe()
# Видим, что есть аномальные значения такие, как tree_dbh - Диаметр дерева, измеренный примерно на высоте 54 дюйма / 137 см над землей

# В целом данный датасет хороший
# Ящик с усами (диаграмма размаха)
f, axes = plt.subplots(2, 1)

sns.boxplot(data.tree_dbh, palette="PRGn", ax=axes[0])
sns.distplot(data.tree_dbh, ax=axes[1])

f, axes = plt.subplots(2, 1)

sns.boxplot(data.borocode, palette="PRGn", ax=axes[0])
sns.distplot(data.borocode, ax=axes[1])

# Нормализация значений в признаке tree_dbh, тк другие данные в норме
data.loc[data.tree_dbh >= 40, 'tree_dbh'] -=20
data.loc[data.tree_dbh >= 100, 'tree_dbh'] //= 10
data.loc[data.tree_dbh >= 300, 'tree_dbh'] //=15

f, axes = plt.subplots(2, 1)

sns.boxplot(data.tree_dbh, palette="PRGn", ax=axes[0])
sns.distplot(data.tree_dbh, ax=axes[1])

# Посмотрим на корреляции между всеми признаками
data.corr()
# Линейная зависимость между парраметрами здесь нет

# Целевым признаком является - tree_dbh, тк именно по нему нам надо выделить районы с аномально хорошим или плохим состоянием деревьев
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
import warnings
warnings.simplefilter('ignore')
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

dataf = data
dataf.drop(dataf[dataf['tree_dbh'] >= 20].index, inplace=True)
dataf['curb_loc'] = np.where(dataf['curb_loc'] == 'OnCurb', 0, 1)
dataf['status'] = np.where(dataf['status'] == 'Alive', 1, 0)
dataf['steward'] = np.where(dataf['steward'] == 'None', 0, 1)
dataf['guards'] = np.where(dataf['guards'] == 'None', 0, 1)
dataf['sidewalk'] = np.where(dataf['sidewalk'] == 'NoDamage', 1, 0)
dataf['problems'] = np.where(dataf['problems'] == 'None', 1, 0) 
dataf['root_stone'] = np.where(dataf['root_stone'] == 'None', 1, 0)
dataf['root_grate'] = np.where(dataf['root_grate'] == 'None', 1, 0)
dataf['root_other'] = np.where(dataf['root_other'] == 'None', 1, 0)
dataf['trunk_wire'] = np.where(dataf['trunk_wire'] == 'None', 1, 0)
dataf['trnk_light'] = np.where(dataf['trnk_light'] == 'None', 1, 0)
dataf['trnk_other'] = np.where(dataf['trnk_other'] == 'None', 1, 0)
dataf['brch_light'] = np.where(dataf['brch_light'] == 'None', 1, 0)
dataf['brch_shoe'] = np.where(dataf['brch_shoe'] == 'None', 1, 0)
dataf['brch_other'] = np.where(dataf['brch_other'] == 'None', 1, 0)

dataf['tree_dbh'].loc[(dataf['tree_dbh'] < 20)] = 0
dataf['tree_dbh'].loc[(dataf['tree_dbh'] >= 20)] = 1
dataf['tree_dbh'].loc[(dataf['tree_dbh'] < 5)] = -1
dataf = dataf.astype(str)

X = dataf.drop(['borough','block_id','spc_latin','spc_common','address','postcode','zip_city','community board','cncldist','st_assem','st_senate','nta','nta_name','health', 'boro_ct','latitude','longitude','x_sp','y_sp' ], axis = 1)

stand_X = pd.DataFrame(preprocessing.scale(X), columns = X.columns)
stand_X = pd.isnull(stand_X)

from sklearn.decomposition import PCA, KernelPCA
for i in [1,2,3,4]:
    pca = PCA(n_components=i)
    pca.fit(stand_X)
    print( pca.explained_variance_ratio_)

# Достаточно 2 признаков для объяснения 90% дисперсии
# Первый признак вносит наибольший вклад в первую компоненту
len(stand_X)

pca = PCA(n_components=2)
pca.fit(stand_X)
X3 = pca.transform(stand_X)
X3
df3 = pd.DataFrame(data=X3, columns=["PC1","PC2"])

df3['Y'] = 0
df3 = df3.astype(float)
for i in range (5848):
    df3.at[i,'Y'] = data['tree_dbh'].iloc[i]
df3.head()

plt.scatter('PC1', 'PC2', c='Y', cmap = 'Set1', data=df3)

pca = PCA(n_components=2)
pca.fit(stand_X)
X3 = pca.transform(stand_X)
X3
df3 = pd.DataFrame(data=X3, columns=["PC1", "PC2"])
df3['Y'] = 0

dataf['tree_dbh'] = dataf['tree_dbh'].astype('float')
df3 = df3.astype(float)
for i in range (5848):
    df3.at[i,'Y'] = dataf['tree_dbh'].iloc[i]
df3.head()

from sklearn import tree
from sklearn.metrics import f1_score
df3 = df3.astype(str)
target = pd.DataFrame(df3['Y']) 
train = pd.DataFrame(df3.drop(['Y'], axis = 1))
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.3, train_size = 0.7, random_state = 42) 
X_train.shape
clf = tree.DecisionTreeClassifier(max_depth=10, random_state=42)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
f1_score(y_test, y_pred, average='macro') 





























