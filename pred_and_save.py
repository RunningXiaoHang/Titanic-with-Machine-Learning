import joblib
#加载模型
rfc_clf = joblib.load('./version/model_1.pkl')
#导入所需的库
import pandas as pd
import numpy as np
#加载数据
pred_data = pd.read_csv('./test.csv')
pred_data['Age'] = pred_data['Age'].fillna(pred_data['Age'].mean())
pred_data['Sex'] = pred_data['Sex'].astype('category').cat.codes
pred_data['Fare'] = pred_data['Fare'].fillna(pred_data['Fare'].mean())
#获取乘客ID
passenger_number = pred_data['PassengerId']
passenger_number = np.array(passenger_number)
#提取特征
X_new = pred_data[["Pclass", "Age", "Sex", "SibSp", "Parch", "Fare"]]
X_new = np.array(X_new)
X_new.astype(np.float_)
#预测
y_pred = rfc_clf.predict(X_new)
y_pred = np.array(y_pred)
#保存数据
finally_data = {"PassengerId": passenger_number,
                "Survived": y_pred}
finally_data_df = pd.DataFrame(finally_data)
finally_data_df.to_csv("./submission.csv")