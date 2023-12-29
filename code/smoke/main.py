
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

# 读取数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 删除'id'列
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# 特征重要性
X_train = train.drop('smoking', axis=1)
y_train = train['smoking']

# 创建随机森林分类器
model = RandomForestClassifier(random_state=42)

# 定义要搜索的参数网格
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700],
    'max_depth': [None, 10, 20, 30, 40, 50, 60],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# 创建网格搜索对象
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)

# 使用网格搜索找到最佳参数
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f"最佳参数: {grid_search.best_params_}")

# 使用最佳参数创建新的模型
best_model = grid_search.best_estimator_

# 训练模型
best_model.fit(X_train, y_train)
print("训练成功")

# 预测
predictions = best_model.predict(test)

# 将预测结果写入文件
np.savetxt('answer.txt', predictions, fmt='%s')  # 使用numpy的savetxt方法可以更简洁地将数组写入文件
