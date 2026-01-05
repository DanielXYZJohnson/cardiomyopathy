"""
基于7项临床指标，预测患者的心肌病类别（CA / HCM / HHD）。

输入 (patient_data):
------------------
- 支持以下两种格式：
    1. **字典 (dict)**：表示单个患者的7个特征，例如：
        {
            'Systolic Blood Pressure': 120,
            'Sokolow Index': 2.1,
            'Interventricular Septum Thickness': 15.0,
            'LV Posterior Wall Thickness': 12.0,
            'TAPSE': 18.0,
            "Average E/E'": 12.0,
            'LVEF': 55.0
        }
    2. **pandas DataFrame**：每行代表一个患者，必须包含上述7列（顺序无关）。

- **允许缺失值（NaN）**：若某些指标缺失（如未测量），函数会使用多重插补（MICE）自动填充，
    基于其他特征和随机森林回归模型进行合理估计，无需用户手动处理缺失。

输出:
-----
- **仅返回最可能的心肌病类别**，类型为字符串，取值为以下之一：
    • 'CA'  —— 心脏淀粉样变性 (Cardiac Amyloidosis)
    • 'HCM' —— 肥厚型心肌病 (Hypertrophic Cardiomyopathy)
    • 'HHD' —— 高血压性心脏病 (Hypertensive Heart Disease)

注意:
----
- 本函数内部会加载预训练的 SuperLearner 集成模型（需已保存在 ./models/super_learner_model.joblib）。
- 若模型加载失败或输入格式错误，返回 None。
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# 1. 定义SuperLearner类 (按您提供的版本)
class SuperLearner(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, n_splits=5, random_state=42):
        self.base_models = base_models
        self.n_splits = n_splits
        self.random_state = random_state
        
    def fit(self, X, y):
        # 确保y是数值类型
        y_values = y.values if hasattr(y, 'values') else y
        if hasattr(y_values, 'codes'):
            y_values = y_values.codes
            
        self.n_classes_ = len(np.unique(y_values))
        self.base_models_fitted_ = []
        
        # 使用交叉验证生成out-of-fold预测
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        meta_features = np.zeros((X.shape[0], len(self.base_models), self.n_classes_))
        
        # 训练基础模型并生成out-of-fold预测
        for i, model in enumerate(self.base_models):
            self.base_models_fitted_.append([])
            temp_preds = np.zeros((X.shape[0], self.n_classes_))
            
            for train_idx, val_idx in kf.split(X):
                # 准备数据
                X_train = X.iloc[train_idx].values if hasattr(X, 'iloc') else X[train_idx]
                X_val = X.iloc[val_idx].values if hasattr(X, 'iloc') else X[val_idx]
                y_train = y_values[train_idx]
                
                # 训练模型
                model_copy = clone(model)
                model_copy.fit(X_train, y_train)
                self.base_models_fitted_[-1].append(model_copy)
                
                # 生成out-of-fold预测
                temp_preds[val_idx] = model_copy.predict_proba(X_val)
            
            meta_features[:, i, :] = temp_preds
        
        # 优化权重
        def loss_function(weights):
            # 确保权重和为1且非负
            weights = np.abs(weights)
            weights = weights / np.sum(weights)
            
            # 计算加权预测
            weighted_preds = np.zeros((X.shape[0], self.n_classes_))
            for i in range(len(self.base_models)):
                weighted_preds += weights[i] * meta_features[:, i, :]
            
            # 计算交叉熵损失
            log_loss = -np.mean([np.log(weighted_preds[i, y_values[i]] + 1e-10) 
                               for i in range(len(y_values))])
            return log_loss
        
        # 优化权重
        n_models = len(self.base_models)
        initial_weights = np.ones(n_models) / n_models
        
        result = minimize(loss_function, initial_weights, method='Nelder-Mead',
                        options={'maxiter': 1000})
        
        # 标准化权重并保存
        self.weights_ = np.abs(result.x)
        self.weights_ = self.weights_ / np.sum(self.weights_)
        
        return self
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, X):
        # 生成每个基础模型的预测
        X_array = X.values if hasattr(X, 'values') else X
        all_predictions = []
        
        # 对每个基础模型集合进行预测
        for models in self.base_models_fitted_:
            model_preds = []
            for model in models:
                model_preds.append(model.predict_proba(X_array))
            # 对每个模型集合的预测取平均
            avg_pred = np.mean(model_preds, axis=0)
            all_predictions.append(avg_pred)
        
        # 计算加权预测
        weighted_predictions = np.zeros((X_array.shape[0], self.n_classes_))
        for i, pred in enumerate(all_predictions):
            weighted_predictions += self.weights_[i] * pred
            
        return weighted_predictions

# 2. 数据预处理函数
def preprocess_patient_data(patient_data):
    """
    预处理病人数据，处理缺失值
    """
    # 确保输入是DataFrame
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    else:
        df = patient_data.copy()
    
    # 确保列顺序正确
    feature_names = ['Systolic Blood Pressure', 'Sokolow Index', 
                     'Interventricular Septum Thickness', 'LV Posterior Wall Thickness',
                     'TAPSE', "Average E/E'", 'LVEF']
    
    # 只保留需要的列
    df = df[feature_names]
    
    # 处理缺失值 - 使用多重插补
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(random_state=42),
        max_iter=10,
        random_state=42
    )
    
    # 拟合并转换数据
    imputed_data = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(imputed_data, columns=feature_names)
    
    return df_imputed

# 3. 加载预训练模型
def load_trained_super_learner():
    """
    从文件加载预训练的SuperLearner模型
    """
    try:
        # 假设模型已保存为super_learner_model.joblib
        model = joblib.load('./models/super_learner_model.joblib')
        return model
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("请确保模型文件存在于正确路径中")
        return None

# 4. 预测函数
def predict_cardiomyopathy(patient_data):
    """
    预测心肌病类型
    """
    # 加载模型
    super_learner = load_trained_super_learner()
    if super_learner is None:
        return None, None
    
    # 预处理数据
    processed_data = preprocess_patient_data(patient_data)
    
    # 预测概率
    probabilities = super_learner.predict_proba(processed_data)[0]
    
    # 预测类别
    prediction = super_learner.predict(processed_data)[0]
    
    return prediction, probabilities

# 5. 简化输出格式
def print_prediction_results(prediction, probabilities):
    """
    简化输出格式，只显示概率和最终预测
    """
    class_names = ['CA', 'HCM', 'HHD']
    max_prob_index = np.argmax(probabilities)
    
    print("\n各类别概率:")
    for i, prob in enumerate(probabilities):
        if i == max_prob_index:
            print(f"- {class_names[i]}: {prob:.4f} (最高概率)")
        else:
            print(f"- {class_names[i]}: {prob:.4f}")
    
    print(f"\n最终预测类别: {class_names[prediction]}")

# 6. 使用示例
if __name__ == "__main__":
    # 示例病人数据
    example_patient = {
        'Systolic Blood Pressure': 120,
        'Sokolow Index': 2.1,
        'Interventricular Septum Thickness': 15.0,
        'LV Posterior Wall Thickness': 12.0,
        'TAPSE': 18.0,
        "Average E/E'": 12.0,
        'LVEF': 55.0
    }
    
    # 获取预测结果
    prediction, probabilities = predict_cardiomyopathy(example_patient)
    
    # 打印简化结果
    if prediction is not None and probabilities is not None:
        print_prediction_results(prediction, probabilities)
    else:
        print("预测失败，请检查模型和输入数据")