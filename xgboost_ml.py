import random, os, glob, math, pathlib, csv, zipfile, warnings

import numpy as np 
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', family='AppleGothic') # 한글 폰트 설정
warnings.filterwarnings(action='ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

from os import listdir
from PIL import Image
from tqdm import tqdm 
from pathlib import Path
from skimage import io

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


base_path = get_ipython().getoutput('pwd')
sub_df = pd.read_csv(".../subject-info.csv", low_memory=False)
test_measure_df = pd.read_csv(".../test_measure.csv", low_memory=False)
# 데이터 결합 (Merge)
df = pd.merge(sub_df, test_measure_df, on='ID_test')
df.dropna(subset=['VO2'], inplace=True) 
cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()
device = torch.device("cuda" if cuda_available else "mps")

for column in df.columns:
    print(f"Unique values in {column}: {df[column].nunique()}")

import matplotlib.pyplot as plt
import seaborn as sns

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
for column in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df_cleaned)
categorical_features = ['Sex']
categorical_indices = [df_cleaned.columns.get_loc(col) for col in categorical_features]
one_hot_encoder = OneHotEncoder(categories='auto', sparse_output=False)
df_encoded_categorical = one_hot_encoder.fit_transform(df_imputed[:, categorical_indices])
scaler = StandardScaler()
numerical_features = [col for col in df_cleaned.columns if col not in categorical_features]
df_scaled_numerical = scaler.fit_transform(df_imputed[:, [df_cleaned.columns.get_loc(col) for col in numerical_features]])
df_processed = np.concatenate((df_encoded_categorical, df_scaled_numerical), axis=1)
target_variable = df_cleaned['VO2']
X_train, X_test, y_train, y_test = train_test_split(df_processed, target_variable, test_size=0.2, random_state=42)

data = df_cleaned.copy()
data['VO2_ml_kg_min'] = data['VO2'] / data['Weight']
data['Age'] = data['Age'].apply(math.ceil)
unique_individuals = train_data['ID_x'].nunique()
print("Number of unique individuals:", unique_individuals)
gender_ratio = train_data['Sex'].value_counts(normalize=True) * 100
print("\nGender ratio:")
print(gender_ratio)
age_ratio_by_gender = train_data.groupby('Sex')['Age'].mean()
print("\nAge ratio by gender:")
print(age_ratio_by_gender)
height_weight_by_gender = train_data.groupby('Sex')[['Height', 'Weight']].mean()
print("\nHeight and weight by gender:")
print(height_weight_by_gender)
vo2_by_gender = train_data.groupby('Sex')[['VO2', 'VO2_ml_kg_min']].mean()
print("\nVO2 max and min by gender:")
print(vo2_by_gender)
age_std_by_gender = train_data.groupby('Sex')['Age'].std()
print("\nAge standard deviation by gender:")
print(age_std_by_gender)
hw_std_by_gender = train_data.groupby('Sex')[['Height', 'Weight']].std()
print("\nHeight and weight standard deviation by gender:")
print(hw_std_by_gender)
import numpy as np

excluded_rows = 0
male_criteria = {
    'Superior': [55.4, 54, 52.5, 48.9, 45.7, 42.1],
    'Excellent': [51.1, 48.3, 46.4, 43.4, 39.5, 36.7],
    'Good': [45.4, 44, 42.4, 39.2, 35.5, 32.3],
    'Fair': [41.7, 40.5, 38.5, 35.6, 32.3, 29.4]
}
female_criteria = {
    'Superior': [49.6, 47.4, 45.3, 41.1, 37.8, 36.7],
    'Excellent': [43.9, 42.4, 39.7, 36.7, 33, 30.9],
    'Good': [39.5, 37.8, 36.3, 33, 30, 28.1],
    'Fair': [36.1, 34.4, 33, 30.1, 27.5, 25.9]
}

def vo2max_category(row):
    global excluded_rows  
    
    age = row['Age']
    sex = row['Sex']
    vo2_ml_kg_min = row['VO2_ml_kg_min']
    
    # 나이가 19세 이하 또는 80세 이상인 행 제외
    if age <= 19 or age >= 80:
        excluded_rows += 1
        return None
    
    criteria = male_criteria if sex == 0 else female_criteria
    
    for category, values in criteria.items():
        if vo2_ml_kg_min >= values[0]:
            return category
    
    return "Poor"

data['VO2max_category'] = data.apply(vo2max_category, axis=1)
print("제외된 행의 수:", excluded_rows)

filtered_data_anomaly = data[~data['VO2max_category'].isin(['Poor', 'Fair', 'Good', 'Excellent', 'Superior'])]
unique_counts = filtered_data_anomaly['ID_x'].value_counts()
age_distribution = filtered_data_anomaly.groupby('Age').size()
print("Unique counts by ID_x:")
print(unique_counts)

print("\nAge distribution:")
print(age_distribution)

plt.figure(figsize=(10, 6))
plt.bar(age_distribution.index, age_distribution.values, color='skyblue')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution of Anomalies')
plt.grid(True)
plt.show()

filtered_data_valid = data[data['VO2max_category'].isin(['Poor', 'Fair', 'Good', 'Excellent', 'Superior'])]
category_counts = filtered_data_valid['VO2max_category'].value_counts()
print("Category Counts:")
print(category_counts)

plt.figure(figsize=(8, 6))
category_counts.plot(kind='bar', color='skyblue')
plt.xlabel('VO2max Category')
plt.ylabel('Count')
plt.title('Distribution of VO2max Categories')
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


data = filtered_data_valid.copy()
min_VO2 = data['VO2_ml_kg_min'].min()
max_VO2 = data['VO2_ml_kg_min'].max()

print("Minimum VO2:", min_VO2)
print("Maximum VO2:", max_VO2)
data = data[data['VO2_ml_kg_min'] >= 0]
data = data.dropna(subset=['VO2_ml_kg_min'])
min_VO2 = data['VO2_ml_kg_min'].min()
max_VO2 = data['VO2_ml_kg_min'].max()
print("Minimum VO2:", min_VO2)
print("Maximum VO2:", max_VO2)
data['VO2_ml_kg_min'] = data['VO2_ml_kg_min'].round(1)
print(data.shape)
data = data.dropna(subset=['VO2_ml_kg_min'])
min_VO2 = data['VO2_ml_kg_min'].min()
max_VO2 = data['VO2_ml_kg_min'].max()
print("Minimum VO2:", min_VO2)
print("Maximum VO2:", max_VO2)

from sklearn.model_selection import train_test_split
unique_ids = data['ID_x'].unique()
train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
train_data = data[~data['ID_x'].isin(test_ids)]
test_data = data[data['ID_x'].isin(test_ids)]
unique_ids_1 = set(train_data['ID_x'])
unique_ids_2 = set(test_data['ID_x'])
overlap_ids = unique_ids_1.intersection(unique_ids_2)



import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = data[['Age', 'Weight', 'Height', 'HR']]
y = data['VO2_ml_kg_min']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 데이터를 텐서로 변환
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

train_data = train_data[['Age', 'Weight', 'Height', 'HR', 'Sex', 'time', 'VO2', 'VO2max_category','VO2_ml_kg_min']]
train_data

from pycaret.regression import *

# PyCaret 설정
exp = setup(data=train_data, target='VO2_ml_kg_min', 
            session_id=77, transformation= True, fold=5, 
            numeric_features=['Age', 'Weight', 'Height', 'HR', 'Sex', 'time'], 
            remove_multicollinearity = True, ## 다중공선성 제거
            multicollinearity_threshold = 0.90)#,
            #pca = True)

from sklearn.model_selection import train_test_split
from pycaret.regression import setup, compare_models, tune_model, blend_models, evaluate_model, automl, predict_model


# 1) Define problem
train_full_data, test_data = train_test_split(data)  # data: pandas.DataFrame
numerical_features = ['X1', 'X2']
nominal_features   = ['X3', 'X4']
ordinal_features   = {'X5' : ['low', 'medium', 'high'], 'X6': ['freshman', 'sophomore', 'junior', 'senior']}.
target             = 'Y'

# 2) Setup PyCaret session
s = setup(train_full_data, target,
          numeric_features=numeric_features,
          categorical_features=nominal_features,
          ordinal_features=ordinal_features,
          use_gpu=True)
base_models = compare_models(n_select=5, sort='MAE')  
tuned_models = [tune_model(model, optimize='MAE', choose_better=True, return_train_score=True) for model in base_models]
ensemble_model = blend_models(tuned_models, choose_better=True, optimize='MAE', return_train_score=True)  # voting(average)
final_model = automl(optimize='MAE', return_train_score=True)  # 생성한 모델들 중 최적의 모델을 선택
evaluate_model(final_model)  # 다양한 평가지표들과 시각자료들을 통해 모델을 평가
preds = predict_model(final_model, data=test_data)  # 'Label' column으로 예측값을 추가

best_model = compare_models(n_select = 3, sort='MAPE', fold=5)

xgboost = create_model('xgboost', fold = 5)
#dt = create_model('dt', fold = 10)
from pycaret.regression import tune_model

tuned_model = tune_model(xgboost,fold=5, 
                         optimize='MAPE', 
                         choose_better = True,
                         return_train_score = True)

save_model(tuned_model, '/Users/daeeon/Desktop/VO2peak-prediction-main/20240423_best_tuned_xgb_VO2max_model')
