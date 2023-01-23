import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#加载数据集
data = pd.read_csv('./train.csv')
data['Sex'] = data['Sex'].astype('category').cat.codes
data['Age'] = data['Age'].fillna(data['Age'].mean())
#提取特征
X_data = data[["Pclass", "Age", "Sex", "SibSp", "Parch", "Fare"]]
y_data = data['Survived']
#划分数据集
X_data, y_data = np.array(X_data), np.array(y_data)
X_data, y_data = X_data.astype(np.float_), y_data.astype(np.int_)
X_train, y_train, X_test, y_test = X_data[0:800], y_data[0:800], X_data[800:-1], y_data[800:-1]
#拟合数据
import joblib
from sklearn.ensemble import RandomForestClassifier
#20次拟合, 取最好结果
for step in range(20):
    step += 1
    rfc = RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=4)
    rfc.fit(X_train, y_train)
    print('epoch: {} score: {}'.format(step, rfc.score(X_test, y_test)), end=' ')
    try:
        #保存模型
        joblib.dump(rfc, './version/model_{}.pkl'.format(step))
        print("model was already saved")
    except:
        print("error")
        



