#!/usr/bin/env python
# coding: utf-8

# # 00 Library & Pathing

# In[1]:


#!pip install --ignore-installed pycaret
# https://blog.naver.com/qkrdnjsrl0628/222791254831
# !pip install pycaret --ignore-installed llvmlite
#! pip install --pre pycaret
# !pip install pycaret==2.3.2
#w/o internet
#!pip download pycaret==2.3.2
#!python -m pip install --find-links=./ pycaret==2.3.2
# !pip install pycaret==2.3.10 markupsafe==2.0.1 pyyaml==5.4.1 -qq


# In[287]:


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


# In[289]:


base_path = get_ipython().getoutput('pwd')
base_path


# In[291]:


sub_df = pd.read_csv("/Users/s-alpha/Desktop/VO2peak-prediction-main/treadmill-maximal-exercise-tests-from-the-exercise-physiology-and-human-performance-lab-of-the-university-of-malaga-1.0.1/subject-info.csv", low_memory=False)
test_measure_df = pd.read_csv("/Users/s-alpha/Desktop/VO2peak-prediction-main/treadmill-maximal-exercise-tests-from-the-exercise-physiology-and-human-performance-lab-of-the-university-of-malaga-1.0.1/test_measure.csv", low_memory=False)


# In[293]:


sub_df.ID.nunique()


# ## 01 AUTOML

# In[300]:


import torch
import pandas as pd
from pycaret.regression import *


sub_df = pd.read_csv("/Users/s-alpha/Desktop/VO2peak-prediction-main/treadmill-maximal-exercise-tests-from-the-exercise-physiology-and-human-performance-lab-of-the-university-of-malaga-1.0.1/subject-info.csv", low_memory=False)
test_measure_df = pd.read_csv("/Users/s-alpha/Desktop/VO2peak-prediction-main/treadmill-maximal-exercise-tests-from-the-exercise-physiology-and-human-performance-lab-of-the-university-of-malaga-1.0.1/test_measure.csv", low_memory=False)


# 데이터 결합 (Merge)
df = pd.merge(sub_df, test_measure_df, on='ID_test')
df.dropna(subset=['VO2'], inplace=True) 

# GPU 및 MPS 사용 가능 여부 확인
cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()
device = torch.device("cuda" if cuda_available else "mps")
print("GPU 사용 가능 여부:", "사용 가능" if cuda_available else "사용 불가능")
print("MPS 사용 가능 여부:", "사용 가능" if mps_available else "사용 불가능")


# ## 01.01 EDA

# In[303]:


# 데이터프레임 기본 정보 출력
print(df.head())
print(df.info())
print(df.describe())

# 결측치 확인
print(df.isnull().sum())

# 각 변수별 unique 값 확인
for column in df.columns:
    print(f"Unique values in {column}: {df[column].nunique()}")

# 각 변수별 분포 확인
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

# 상관관계 확인
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[92]:


# 성별에 따른 VO2 분포 시각화
plt.figure(figsize=(8, 6))
sns.boxplot(x='Sex', y='VO2', data=df)
plt.title("Distribution of VO2 by Sex")
plt.xlabel("Sex")
plt.ylabel("VO2")
plt.show()

# 나이에 따른 VO2 분포 시각화
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Age', y='VO2', data=df)
plt.title("Scatter plot of VO2 by Age")
plt.xlabel("Age")
plt.ylabel("VO2")
plt.show()

# 시간에 따른 VO2 분포 시각화
plt.figure(figsize=(10, 8))
sns.lineplot(x='time', y='VO2', data=df)
plt.title("Line plot of VO2 by Time")
plt.xlabel("Time")
plt.ylabel("VO2")
plt.show()

# 심박수(HR)와 VO2의 관계 시각화
plt.figure(figsize=(8, 6))
sns.scatterplot(x='HR', y='VO2', data=df)
plt.title("Scatter plot of VO2 by HR")
plt.xlabel("HR")
plt.ylabel("VO2")
plt.show()


# In[13]:


min_vo2 = df['VO2'].min()
max_vo2 = df['VO2'].max()

print("Minimum VO2:", min_vo2)
print("Maximum VO2:", max_vo2)


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

# 음수인 VO2 값 확인
negative_vo2 = df[df['VO2'] < 0]
print("Negative VO2 count:", len(negative_vo2))

# VO2의 분포 시각화
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='VO2', kde=True)
plt.title('Distribution of VO2')
plt.xlabel('VO2')
plt.ylabel('Frequency')
plt.show()

# Boxplot을 이용한 이상치 시각화
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='VO2')
plt.title('Boxplot of VO2')
plt.xlabel('VO2')
plt.show()


# # 01.01.02 이상치 제거

# In[304]:


from scipy import stats

# Z-score 계산
z_scores = stats.zscore(df['VO2'])

# Z-score가 특정 기준 이상인 데이터 제거
threshold = 3
outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
df_cleaned = df.drop(outlier_indices)

# 이상치 제거 후 데이터 확인
print("Original DataFrame shape:", df.shape)
print("DataFrame shape after removing outliers:", df_cleaned.shape)


# In[305]:


# NaN 값을 가진 행 제거
df_cleaned = df_cleaned.dropna()

# NaN 값을 제거한 후 데이터 확인
print("DataFrame shape after removing NaN values:", df_cleaned.shape)


# In[17]:


# 각 환자별로 몇 번의 ID_test를 생성했는지 카운트
id_test_counts = df_cleaned.groupby('ID_x')['ID_test'].nunique()

# 전체 환자 수
total_patients = df_cleaned['ID_x'].nunique()

# 결과 출력
print("Total number of patients:", total_patients)
print("Number of ID_test per patient:")
print(id_test_counts)


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt

# total_patients 시각화
plt.figure(figsize=(10, 6))
sns.countplot(x='ID_x', data=df_cleaned)
plt.title('Number of Tests per Patient')
plt.xlabel('Number of Patients')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


# ### 03 Preprocessing

# In[309]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

# 누락된 값 처리
imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df_cleaned)

# 범주형 데이터 인덱스 가져오기
categorical_features = ['Sex']
categorical_indices = [df_cleaned.columns.get_loc(col) for col in categorical_features]

# 범주형 데이터를 One-Hot 인코딩
one_hot_encoder = OneHotEncoder(categories='auto', sparse_output=False)
df_encoded_categorical = one_hot_encoder.fit_transform(df_imputed[:, categorical_indices])

# 범주형 데이터를 제외한 수치형 데이터 표준화
scaler = StandardScaler()
numerical_features = [col for col in df_cleaned.columns if col not in categorical_features]
df_scaled_numerical = scaler.fit_transform(df_imputed[:, [df_cleaned.columns.get_loc(col) for col in numerical_features]])

# 인코딩된 데이터와 스케일링된 데이터를 결합
df_processed = np.concatenate((df_encoded_categorical, df_scaled_numerical), axis=1)

# 이상치 처리
# 여기에서 이상치 처리 방법을 추가할 수 있습니다.

# 목표 변수 추출
target_variable = df_cleaned['VO2']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(df_processed, target_variable, test_size=0.2, random_state=42)


# In[310]:


# 데이터 준비
data = df_cleaned.copy()


# In[313]:


# data['VO2']*(220-data['Age']-73-(data['Sex']*10))/(data['HR']-73-(data['Sex']*10))
data['VO2_ml_kg_min'] = data['VO2'] / data['Weight']
data


# In[315]:


data['Age'] = data['Age'].apply(math.ceil)
data


# In[319]:


import pandas as pd

# Unique individuals based on 'ID_x'
unique_individuals = train_data['ID_x'].nunique()
print("Number of unique individuals:", unique_individuals)

# Gender ratio (male vs. female)
gender_ratio = train_data['Sex'].value_counts(normalize=True) * 100
print("\nGender ratio:")
print(gender_ratio)

# Age ratio by gender
age_ratio_by_gender = train_data.groupby('Sex')['Age'].mean()
print("\nAge ratio by gender:")
print(age_ratio_by_gender)

# Height and weight by gender
height_weight_by_gender = train_data.groupby('Sex')[['Height', 'Weight']].mean()
print("\nHeight and weight by gender:")
print(height_weight_by_gender)

# VO2 max and min by gender
vo2_by_gender = train_data.groupby('Sex')[['VO2', 'VO2_ml_kg_min']].mean()
print("\nVO2 max and min by gender:")
print(vo2_by_gender)


# In[321]:


# Age standard deviation by gender
age_std_by_gender = train_data.groupby('Sex')['Age'].std()
print("\nAge standard deviation by gender:")
print(age_std_by_gender)


# In[323]:


# Height and weight standard deviation by gender
hw_std_by_gender = train_data.groupby('Sex')[['Height', 'Weight']].std()
print("\nHeight and weight standard deviation by gender:")
print(hw_std_by_gender)


# In[ ]:





# In[325]:


import numpy as np

# https://www8.garmin.com/manuals/webhelp/venu/EN-US/Venu_OM_EN-US.pdf

# 전역 변수로 제외된 행의 수를 초기화
excluded_rows = 0

# 주어진 기준에 따라 숫자 변경
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

# VO2max 범주화 함수 정의
def vo2max_category(row):
    global excluded_rows  # 전역 변수로 사용할 것임을 명시
    
    age = row['Age']
    sex = row['Sex']
    vo2_ml_kg_min = row['VO2_ml_kg_min']
    
    # 나이가 19세 이하 또는 80세 이상인 행 제외
    if age <= 19 or age >= 80:
        excluded_rows += 1
        return None
    
    # 성별에 따라 기준 선택
    criteria = male_criteria if sex == 0 else female_criteria
    
    # 기준에 따라 범주화
    for category, values in criteria.items():
        if vo2_ml_kg_min >= values[0]:
            return category
    
    # 범주가 없는 경우 "Poor" 반환
    return "Poor"

# 범주 적용
data['VO2max_category'] = data.apply(vo2max_category, axis=1)

# 제외된 행의 수 출력
print("제외된 행의 수:", excluded_rows)



# In[326]:


data


# In[329]:


# 'Poor', 'Fair', 'Good', 'Excellent', 'Superior' 이 아닌 값을 가진 행 추출
filtered_data_anomaly = data[~data['VO2max_category'].isin(['Poor', 'Fair', 'Good', 'Excellent', 'Superior'])]

# 'ID_x'를 기준으로 고유 갯수 확인
unique_counts = filtered_data_anomaly['ID_x'].value_counts()

# 연령별 분산도 확인
age_distribution = filtered_data_anomaly.groupby('Age').size()

# 결과 출력
print("Unique counts by ID_x:")
print(unique_counts)

print("\nAge distribution:")
print(age_distribution)

# 연령별 분산도 시각화
plt.figure(figsize=(10, 6))
plt.bar(age_distribution.index, age_distribution.values, color='skyblue')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution of Anomalies')
plt.grid(True)
plt.show()


# In[331]:


filtered_data_valid = data[data['VO2max_category'].isin(['Poor', 'Fair', 'Good', 'Excellent', 'Superior'])]
filtered_data_valid


# In[334]:


df.ID_x.nunique()


# In[76]:


filtered_data_anomaly.ID_y.nunique()


# In[56]:


filtered_data_anomaly


# In[336]:


filtered_data_valid.ID_x.nunique()


# In[338]:


filtered_data_valid


# In[340]:


# 'VO2max_category' 열의 분포 확인
category_counts = filtered_data_valid['VO2max_category'].value_counts()

# 결과 출력
print("Category Counts:")
print(category_counts)

# 분포 시각화
plt.figure(figsize=(8, 6))
category_counts.plot(kind='bar', color='skyblue')
plt.xlabel('VO2max Category')
plt.ylabel('Count')
plt.title('Distribution of VO2max Categories')
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[342]:


data = filtered_data_valid.copy()
data


# In[344]:


data.ID_y.nunique()


# In[346]:


min_VO2 = data['VO2_ml_kg_min'].min()
max_VO2 = data['VO2_ml_kg_min'].max()

print("Minimum VO2:", min_VO2)
print("Maximum VO2:", max_VO2)


# In[348]:


# 음수 값 제거
data = data[data['VO2_ml_kg_min'] >= 0]

# NaN 값 제거
data = data.dropna(subset=['VO2_ml_kg_min'])

min_VO2 = data['VO2_ml_kg_min'].min()
max_VO2 = data['VO2_ml_kg_min'].max()

print("Minimum VO2:", min_VO2)
print("Maximum VO2:", max_VO2)


# In[350]:


data['VO2_ml_kg_min'] = data['VO2_ml_kg_min'].round(1)
print(data.shape)


# In[352]:


data = data.dropna(subset=['VO2_ml_kg_min'])

min_VO2 = data['VO2_ml_kg_min'].min()
max_VO2 = data['VO2_ml_kg_min'].max()

print("Minimum VO2:", min_VO2)
print("Maximum VO2:", max_VO2)


# # 00.04 Patients Overlap

# In[356]:


from sklearn.model_selection import train_test_split

# data.ID_x를 기준으로 고유한 값들을 추출합니다.
unique_ids = data['ID_x'].unique()

# 고유한 값들을 무작위로 섞은 후, 20%를 테스트 세트로 선택합니다.
train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

# 테스트 세트의 data.ID_x 값을 기준으로 해당하는 데이터를 테스트 세트로 분리하고, 나머지는 훈련 세트로 분리합니다.
train_data = data[~data['ID_x'].isin(test_ids)]
test_data = data[data['ID_x'].isin(test_ids)]

# 결과 확인
print("훈련 데이터 크기:", len(train_data))
print("테스트 데이터 크기:", len(test_data))


# In[358]:


# 첫 번째 데이터 프레임의 고유한 ID_x 값들을 추출합니다.
unique_ids_1 = set(train_data['ID_x'])

# 두 번째 데이터 프레임의 고유한 ID_x 값들을 추출합니다.
unique_ids_2 = set(test_data['ID_x'])

# 두 데이터 프레임의 ID_x 값들을 비교하여 중복되는 값이 있는지 확인합니다.
overlap_ids = unique_ids_1.intersection(unique_ids_2)

if overlap_ids:
    print("두 데이터의 ID_x에 중복되는 값이 있습니다.")
    print("중복되는 ID_x 값들:", overlap_ids)
else:
    print("두 데이터의 ID_x에 중복되는 값이 없습니다.")


# In[360]:


train_data.ID_x.nunique()


# In[362]:


test_data.ID_x.nunique()


# ## [X] 01.03 Deep Learning

# In[147]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[172]:


# 특성과 타겟 분리
X = data[['Age', 'Weight', 'Height', 'HR']]
y = data['VO2_ml_kg_min']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 데이터를 텐서로 변환
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)


# ### 모델 앙상블

# In[217]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

# 모델 설계
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 모델 학습 함수 정의
def train_model_with_random_forest(model, X_train, y_train, X_test, y_test, device):
    model.to(device)
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32).to(device)

    # 모델 훈련
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss() #MSELoss()
    num_epochs = 100

    # Progress bar 생성
    progress_bar = tqdm(total=num_epochs, desc='Training', position=0)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Progress bar 업데이트
        progress_bar.set_postfix({'Loss': loss.item()})
        progress_bar.update()

    # Random Forest 모델 초기화 및 학습
    random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
    random_forest.fit(X_train, y_train)

    # 모델 성능 평가
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        mape_nn = mean_absolute_percentage_error(y_test_tensor.cpu().numpy(), y_pred.cpu().numpy())

    # Random Forest 성능 평가
    y_pred_rf = random_forest.predict(X_test)
    mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)

    print(f'Mean Absolute Percentage Error (Neural Network): {mape_nn:.4f}')
    print(f'Mean Absolute Percentage Error (Random Forest): {mape_rf:.4f}')

# MAPE 계산 함수 정의
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 데이터 전처리
X = data[['Age', 'Weight', 'Height', 'HR']]
y = data['VO2_ml_kg_min']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 학습 및 성능 평가
model = RegressionModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model_with_random_forest(model, X_train, y_train, X_test, y_test, device)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 01.02 AUTOML

# https://velog.io/@ddangchani/%EB%94%B0%EB%A6%89%EC%9D%B4-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%84%EC%84%9D%ED%95%98%EA%B8%B0-7-AutoML

# In[364]:


train_data = train_data[['Age', 'Weight', 'Height', 'HR', 'Sex', 'time', 'VO2', 'VO2max_category','VO2_ml_kg_min']]
train_data


# In[366]:


from pycaret.regression import *

# PyCaret 설정
exp = setup(data=train_data, target='VO2_ml_kg_min', 
            session_id=77, transformation= True, fold=5, 
            numeric_features=['Age', 'Weight', 'Height', 'HR', 'Sex', 'time'], 
            remove_multicollinearity = True, ## 다중공선성 제거
            multicollinearity_threshold = 0.90)#,
            #pca = True)


# > 첫줄에서는 데이터프레임과 타겟 변수(반응변수)를 지정해주었다. 또한, session_id는 사이킷런의 random_state와 같이 학습과정의 randomness를 제어하게끔 해주는 변수이다. 둘째줄의 normalize는 변수의 정규화를 진행할 것인지 설정하는 것인데, 앞서 정규화를 파이프라인으로 처리했지만 여기서도 일단 True로 설정했다. transformation = True는 데이터가 정규분포 형태를 취하도록 로그변환 등을 수행하도록 하는 변수이며, transform_target은 타겟 변수에 대한 정규변환을 의미한다.
# 네번째 줄의 remove_multicolinearity는 다중공선성을 일으키는 예측변수를 제거할 것인지 설정하는 변수인데, 다음 threshold 변수에서 그 기준치를 설정한다(0.9). 이밖에도, pca = True 를 설정하여 차원축소를 진행할 것인지, remove_outliers = True를 설정하여 이상치를 제거할 것인지 등을 별도로 설정할 수 있다.
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

# 3) Compare models
base_models = compare_models(n_select=5, sort='MAE')  # MAE 기준 상위 5개 모델을 선택

# 4) Hyperparameter tuning
tuned_models = [tune_model(model, optimize='MAE', choose_better=True, return_train_score=True) for model in base_models]

# 5) Ensemble models
ensemble_model = blend_models(tuned_models, choose_better=True, optimize='MAE', return_train_score=True)  # voting(average)

# 6) Select final model
final_model = automl(optimize='MAE', return_train_score=True)  # 생성한 모델들 중 최적의 모델을 선택

# 7) Evaluate model
evaluate_model(final_model)  # 다양한 평가지표들과 시각자료들을 통해 모델을 평가

# 8) Predict
preds = predict_model(final_model, data=test_data)  # 'Label' column으로 예측값을 추가
# In[368]:


# 모델 학습 및 비교
# https://velog.io/@ezoo0422/Python-pycaret%EC%9D%84-%EC%82%AC%EC%9A%A9%ED%95%98%EC%97%AC-%EB%AA%A8%EB%8D%B8-%EC%84%A0%EC%A0%95%ED%95%98%EA%B8%B0

best_model = compare_models(n_select = 3, sort='MAPE', fold=5)


# In[369]:


# lar = create_model('lar', fold = 10)
xgboost = create_model('xgboost', fold = 5)
#dt = create_model('dt', fold = 10)

save_model(et, '/Users/s-alpha/Desktop/VO2peak-prediction-main/20240326_best_VO2max_rf_model')

# In[372]:


from pycaret.regression import tune_model

tuned_model = tune_model(xgboost,fold=5, 
                         optimize='MAPE', 
                         choose_better = True,
                         return_train_score = True)


# In[109]:


# 모델 저장
save_model(tuned_model, '/Users/s-alpha/Desktop/VO2peak-prediction-main/20240423_best_tuned_xgb_VO2max_model')


# In[374]:


plot_model(tuned_model, plot="error")


# In[376]:


plot_model(tuned_model, plot="feature")


# # 01.03 Blend Model(모델 앙살블)

# ## 01.03 Save Model

# In[378]:


# 운동 강도 예측
predictions = predict_model(tuned_model, data=data)


# ## 01.04 Model Evaluation

# In[380]:


# 모델 평가
evaluate_model(tuned_model)


# ## 01.05 Model Evaluation

# In[382]:


test_data = test_data[['Age', 'Weight', 'Height', 'HR', 'Sex', 'time','VO2', 'VO2max_category','VO2_ml_kg_min']]
test_data


# In[384]:


test_predictions = predict_model(tuned_model, data = test_data)
test_predictions.head(5)


# In[386]:


df_test = test_predictions

# 'VO2_ml_kg_min' 및 'prediction_label' 열의 값 반올림
df_test['VO2_ml_kg_min'] = df_test['VO2_ml_kg_min'].round()
df_test['prediction_label'] = df_test['prediction_label'].round()
df_test


# In[388]:


# 운동 강도 예측
predictions = predict_model(tuned_model, data=df_test)


# In[394]:


# 모델 평가
evaluate_model(tuned_model)


# In[404]:


plot_model(tuned_model, plot="feature")


# In[409]:


test_predictions


# In[419]:


test_predictions


# In[431]:


import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(x='VO2max_category', y='prediction_label', data=test_predictions)
plt.xlabel('VO2 Max Category')
plt.ylabel('Predicted VO2 Max')
plt.title('Predicted VO2 Max by VO2 Max Category')
plt.show()


# In[443]:


plt.figure(figsize=(10, 6))
sns.violinplot(x='VO2max_category', y='prediction_label', 
               data=test_predictions, palette='Pastel1')
plt.xlabel('Actual VO2 Max Category')
plt.ylabel('Predicted VO2 Max Category')
plt.title('Actual vs Predicted VO2 Max Category Distribution')
plt.grid(True)
plt.show()


# In[457]:


from pycaret.regression import plot_model

# 모델에서 기능 중요도 플로팅
plot_model(tuned_model, plot='feature')


# In[459]:


from pycaret.regression import plot_model

# 모델에서 부트스트랩 샘플링 플로팅
plot_model(tuned_model, plot='error')


# In[465]:


df_test


# In[ ]:





# In[473]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.countplot(x='VO2max_category', data=df_test, palette='pastel')
plt.title('Distribution of VO2 Max Categories')
plt.xlabel('VO2 Max Category')
plt.ylabel('Count')
plt.show()


# In[479]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='prediction_label', y='VO2_ml_kg_min', data=df_test, palette='pastel')
plt.title('Actual vs Predicted VO2')
plt.xlabel('Actual VO2 (ml/kg/min)')
plt.ylabel('Predicted VO2')
plt.show()


# In[483]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='VO2_ml_kg_min', data=df_test, hue='Sex', palette='pastel')
plt.title('Predicted VO2 by Age and Sex')
plt.xlabel('Age')
plt.ylabel('Predicted VO2')
plt.legend(title='Sex', loc='upper right')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[127]:


df_test.to_csv('/Users/s-alpha/Desktop/VO2peak-prediction-main/20240423_best_tuned_xgboost_VO2max_model.csv', 
               index = False)


# In[1]:


import pandas as pd 

df_test = pd.read_csv('/Users/s-alpha/Desktop/VO2peak-prediction-main/20240423_best_tuned_xgboost_VO2max_model.csv')
df_test


# In[175]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


# df_test의 'VO2_ml_kg_min' 열과 'prediction_label' 열을 각각 y_true와 y_pred로 설정
y_true = df_test['VO2_ml_kg_min']
y_pred = df_test['prediction_label']

# 예측값과 실제값 비교하여 정확도 계산
accuracy = accuracy_score(y_true, y_pred)

# 결과 출력
print(f'Accuracy: {accuracy:.4f}')



# In[131]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# confusion matrix 계산
cm = confusion_matrix(y_true, y_pred)

# 시각화
plt.figure(figsize=(32, 24))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[133]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
print("Mean Squared Error:", mse)


# In[135]:


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
print("Mean Absolute Error:", mae)


# In[137]:


from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
print("R-squared:", r2)


# In[139]:


mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print("MAPE:", mape)


# In[141]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix 계산
cm = confusion_matrix(y_true, y_pred)

# 시각화
plt.figure(figsize=(32, 24))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', 
            xticklabels=[f'Predicted {i}' for i in range(1, cm.shape[1] + 1)],
            yticklabels=[f'Actual {i}' for i in range(1, cm.shape[0] + 1)])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[179]:


import pandas as pd 

df_test = pd.read_csv('/Users/s-alpha/Desktop/VO2peak-prediction-main/20240423_best_tuned_xgboost_VO2max_model.csv')
df_test


# In[181]:


df_test['VO2_ml_kg_min'] = df_test['VO2_ml_kg_min'].astype(float)
df_test['Age'] = df_test['Age'].astype(float)
df_test['Weight'] = df_test['Weight'].astype(float)
df_test


# In[183]:


from sklearn.preprocessing import LabelEncoder

# LabelEncoder 객체 생성
label_encoder = LabelEncoder()

# 'VO2max_category' 열을 라벨 인코딩하여 값 덮어씌우기
df_test['VO2max_category'] = label_encoder.fit_transform(df_test['VO2max_category'])

# 변환된 데이터프레임 확인
df_test


# In[185]:


df_test


# In[187]:


# 시간을 기준으로 person 열을 생성합니다.
df_test['person'] = (df_test['time'] == 0).cumsum()

# 시간이 0인 row들의 index를 가져옵니다.
idx = df_test[df_test['time'] == 0].index

# 0 다음에 오는 row들에 대해 person 값을 1씩 증가시킵니다.
for i in range(1, len(idx)):
    df_test.loc[idx[i]:, 'person'] += i

# person 열을 다시 설정합니다.
df_test.reset_index(drop=True, inplace=True)
df_test


# In[195]:


# time이 0일 때마다 person을 구분하는 기준 컬럼 생성
df_test['person'] = (df_test['time'] == 0).cumsum()

# 각 person의 VO2_ml_kg_min peak 확인
peak_VO2_ml_kg_min = df_test.groupby('person').apply(lambda x: x.loc[x['VO2_ml_kg_min'].idxmax()])

peak_VO2_ml_kg_min


# In[209]:


import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot 그리기
plt.figure(figsize=(8, 6))
sns.scatterplot(data=peak_VO2_ml_kg_min, x='VO2_ml_kg_min', y='prediction_label')
plt.title('Scatter Plot of VO2_ml_kg_min vs. Prediction Label')
plt.xlabel('VO2_ml_kg_min')
plt.ylabel('Prediction Label')
plt.grid(True)
plt.show()


# In[211]:


import seaborn as sns
import matplotlib.pyplot as plt

# Actual과 Predicted가 일치하는지 여부에 따라 색상을 다르게 표시
peak_VO2_ml_kg_min['Correct'] = peak_VO2_ml_kg_min['VO2_ml_kg_min'] == peak_VO2_ml_kg_min['prediction_label']
colors = ['red' if correct else 'blue' for correct in peak_VO2_ml_kg_min['Correct']]

# Scatter plot 그리기
plt.figure(figsize=(10, 8))
sns.scatterplot(data=peak_VO2_ml_kg_min, x='VO2_ml_kg_min', y='prediction_label', hue=peak_VO2_ml_kg_min['Correct'], palette=colors, alpha=0.7)
plt.title('Scatter Plot of VO2_ml_kg_min vs. Prediction Label')
plt.xlabel('VO2_ml_kg_min')
plt.ylabel('Prediction Label')
plt.grid(True)
plt.legend(title='Correct')
plt.show()


# In[213]:


import seaborn as sns
import matplotlib.pyplot as plt

# Actual과 Predicted가 일치하는지 여부에 따라 색상을 다르게 표시
peak_VO2_ml_kg_min['Correct'] = peak_VO2_ml_kg_min['VO2_ml_kg_min'] == peak_VO2_ml_kg_min['prediction_label']
colors = ['red' if correct else 'blue' for correct in peak_VO2_ml_kg_min['Correct']]

# Scatter plot 그리기
plt.figure(figsize=(10, 8))
sns.scatterplot(data=peak_VO2_ml_kg_min, x='VO2_ml_kg_min', y='prediction_label', hue=peak_VO2_ml_kg_min['Correct'], palette=colors, alpha=0.7)
plt.title('Scatter Plot of VO2_ml_kg_min vs. Prediction Label')
plt.xlabel('VO2_ml_kg_min')
plt.ylabel('Prediction Label')
plt.grid(True)

# True 값을 가진 점들을 선으로 연결하여 그리기
true_indices = peak_VO2_ml_kg_min[peak_VO2_ml_kg_min['Correct']].index
for idx in range(len(true_indices) - 1):
    plt.plot([peak_VO2_ml_kg_min.loc[true_indices[idx], 'VO2_ml_kg_min'], peak_VO2_ml_kg_min.loc[true_indices[idx + 1], 'VO2_ml_kg_min']], 
             [peak_VO2_ml_kg_min.loc[true_indices[idx], 'prediction_label'], peak_VO2_ml_kg_min.loc[true_indices[idx + 1], 'prediction_label']], 
             color='green', linestyle='--', linewidth=2)

plt.legend(title='Correct')
plt.show()


# In[486]:


df_test


# In[273]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# R squared 값 계산
r_squared = np.corrcoef(peak_VO2_ml_kg_min['VO2_ml_kg_min'], peak_VO2_ml_kg_min['prediction_label'])[0, 1] ** 2

# 산점도 그래프 그리기
plt.figure(figsize=(10, 8))
sns.scatterplot(data=peak_VO2_ml_kg_min, x='VO2_ml_kg_min', y='prediction_label', 
                hue=peak_VO2_ml_kg_min['Correct'], alpha=0.7)

plt.title('Scatter Plot of VO2_ml_kg_min vs. Prediction Label')
plt.xlabel('VO2_ml_kg_min')
plt.ylabel('Prediction Label')
plt.grid(True)


# diagnol 로 그리기
x_values = np.linspace(0, 100)
plt.plot(x_values, x_values, color='gray', linestyle='--')

# R squared 값 표시
plt.text(0.9, 0.1, f'R squared = {r_squared:.4f}', transform=plt.gca().transAxes, fontsize=12, ha='right')

plt.legend(title='Correct')
plt.show()



# In[276]:


peak_VO2_ml_kg_min


# In[278]:


import numpy as np

# Mean Absolute Error (MAE)
mae = np.mean(np.abs(peak_VO2_ml_kg_min['prediction_label'] - peak_VO2_ml_kg_min['VO2_ml_kg_min']))

# Mean Squared Error (MSE)
mse = np.mean((peak_VO2_ml_kg_min['prediction_label'] - peak_VO2_ml_kg_min['VO2_ml_kg_min']) ** 2)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Root Mean Squared Logarithmic Error (RMSLE)
rmsle = np.sqrt(np.mean((np.log(peak_VO2_ml_kg_min['prediction_label'] + 1) - np.log(peak_VO2_ml_kg_min['VO2_ml_kg_min'] + 1)) ** 2))

# Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((peak_VO2_ml_kg_min['VO2_ml_kg_min'] - peak_VO2_ml_kg_min['prediction_label']) / peak_VO2_ml_kg_min['VO2_ml_kg_min'])) * 100

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Root Mean Squared Logarithmic Error (RMSLE):", rmsle)
print("Mean Absolute Percentage Error (MAPE):", mape)


# In[282]:


import numpy as np

# Calculate R-squared
def r_squared(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Compute R-squared
r2 = r_squared(peak_VO2_ml_kg_min['VO2_ml_kg_min'], peak_VO2_ml_kg_min['prediction_label'])

print("R-squared:", r2)


# In[514]:


train_data


# In[534]:


train_data


# In[532]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# correlation matrix 계산
correlation_matrix = train_data[['Age', 'Weight', 'Height', 'HR', 'time', 'VO2']].corr()

# 히트맵으로 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='BuGn', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[560]:


df_test_numeric


# In[ ]:


# 'VO2max_category' 열 제거
df_test_numeric = df_test.drop(columns=['VO2max_category'])
df_test_numeric = df_test.drop(columns=['prediction_label'])


# In[574]:


plt.rcParams['axes.unicode_minus'] = False

# 피어슨 상관 계수 계산
correlation_matrix = df_test_numeric.corr(method='pearson')

# 히트맵 그리기
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='BuGn', fmt=".2f", linewidths=0.5)
plt.title('Pearson Correlation Matrix')
plt.show()


# In[640]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# pearson 상관 계수 계산
corr_matrix = df_test_numeric.corr(method='pearson')

# 상관 계수 행렬의 반만 가져오기
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Heatmap 그리기
plt.figure(figsize=(10, 8))
#sns.heatmap(corr_matrix, annot=True, cmap='BuGn', mask=mask)
sns.heatmap(corr_matrix, mask=mask, cmap='BuGn', annot=True, fmt=".2f", linewidths=0.5)

plt.title('Pearson Correlation Coefficient Matrix')
plt.grid(False)
plt.show()


# In[714]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df_test_numeric is your DataFrame and it's already been imported
corr_matrix = df_test_numeric.corr(method='pearson')

# Create a mask for the upper triangle, including the diagonal
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Draw the heatmap with the mask and correct aspect ratio
ax = sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")

# Add a title to the heatmap
plt.title('Pearson Correlation Coefficient Matrix (Upper Triangle)')

# Move x-axis to the top of the heatmap
ax.xaxis.tick_top()  # move x-axis ticks to the top
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # set x-axis labels rotation

# Adjust the position of the x-axis labels to align more closely with the top of the heatmap
ax.xaxis.set_label_position('top') 
ax.tick_params(axis='x', which='major', pad=-2)  # Reduce padding to make labels closer to axis

# Ensure y-axis labels are readable
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# Show the plot
plt.show()




# In[718]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df_test_numeric is your DataFrame and it's already been imported
# Calculate the Pearson correlation coefficients
corr_matrix = df_test_numeric.corr(method='pearson')

# Create a mask for the upper triangle but include the diagonal (k=1 includes only above diagonal)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, cmap='BuGn', annot=True, fmt=".2f", linewidths=0.5, mask=mask)

# Set the title of the graph
plt.title('Pearson Correlation Coefficient Matrix (Upper Triangle with Ones)')

# Hide the grid
plt.grid(False)

# Display the plot
plt.show()




# In[542]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

# 각 feature와 정답 사이의 피어슨 상관 계수 계산
correlation_matrix = df_test.corr(method='pearson')['prediction_label']

# 각 feature의 feature importance 계산
X = df_test.drop(columns=['prediction_label'])
y = df_test['prediction_label']

# Random Forest Regressor를 사용하여 feature importance 계산
rf_regressor = RandomForestRegressor()
rf_regressor.fit(X, y)
rf_feature_importance = rf_regressor.feature_importances_

# SVR을 사용하여 feature importance 계산
svr_regressor = SVR(kernel='linear')
svr_regressor.fit(X, y)
svr_permutation_importance = permutation_importance(svr_regressor, X, y, n_repeats=30, random_state=0)
svr_feature_importance = svr_permutation_importance.importances_mean

# 결과를 DataFrame으로 변환
result_df = pd.DataFrame({
    'Feature': X.columns,
    'Pearson Correlation': correlation_matrix.values,
    'RF Feature Importance': rf_feature_importance,
    'SVR Feature Importance': svr_feature_importance
})

# 결과를 출력
print(result_df)


# In[548]:


category_mapping


# In[550]:


df_test


# In[ ]:


# VO2max_category 열의 카테고리를 숫자로 매핑
category_mapping = {'Poor': 0, 'Fair': 1, 'Average': 2, 'Good': 3, 'Excellent': 4}
df_test['VO2max_category_mapped'] = df_test['VO2max_category'].map(category_mapping)
df_test = df_test.drop(columns=['VO2max_category'])


# In[558]:


df_test.dropna(inplace=True)

# 피어슨 상관 계수 계산
correlation_matrix = df_test.corr(method='pearson')['prediction_label']

# 각 feature의 feature importance 계산
X = df_test.drop(columns=['prediction_label'])
y = df_test['prediction_label']

# 모델링과 feature importance 계산하는 코드는 이전과 동일하게 사용합니다.

# Random Forest Regressor를 사용하여 feature importance 계산
rf_regressor = RandomForestRegressor()
rf_regressor.fit(X, y)
rf_feature_importance = rf_regressor.feature_importances_

# SVR을 사용하여 feature importance 계산
svr_regressor = SVR(kernel='linear')
svr_regressor.fit(X, y)
svr_permutation_importance = permutation_importance(svr_regressor, X, y, n_repeats=30, random_state=0)
svr_feature_importance = svr_permutation_importance.importances_mean

# 결과를 DataFrame으로 변환
result_df = pd.DataFrame({
    'Feature': X.columns,
    'Pearson Correlation': correlation_matrix.values,
    'RF Feature Importance': rf_feature_importance,
    'SVR Feature Importance': svr_feature_importance
})

# 결과를 출력
print(result_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[113]:


from pycaret.classification import load_model

# 저장된 모델 불러오기
loaded_model = load_model('/Users/s-alpha/Desktop/VO2peak-prediction-main/20240423_best_tuned_xgb_VO2max_model')

# 모델 요약 정보 출력
print(loaded_model)


# In[121]:


loaded_model[-1]


# In[125]:


print(loaded_model[-1])


# In[8]:


loaded_model = loaded_model[-1]
loaded_model


# In[111]:





# In[11]:


import shap

# SHAP 값 계산
explainer = shap.Explainer(loaded_model, df_test.drop(columns=['prediction_label']))


# In[12]:


shap_values = explainer.shap_values(df_test.drop(columns=['prediction_label']), check_additivity=False)


# In[17]:


# SHAP 알고리즘 지정
#shap.initjs()
#shap_values = shap.TreeExplainer(loaded_model, algorithm="auto").shap_values(X_test)


# SHAP summary plot 그리기
shap.summary_plot(shap_values, df_test.drop(columns=['prediction_label']))


# In[19]:


shap.dependence_plot('Age', shap_values, df_test.drop(columns=['prediction_label']))


# In[23]:


shap.force_plot(explainer.expected_value, shap_values[0,:], df_test.drop(columns=['prediction_label']).iloc[0,:])


# In[27]:


# SHAP 값을 계산
shap_values_single = explainer.shap_values(df_test.drop(columns=['prediction_label']).iloc[0])


# In[ ]:


# Explanation 객체 생성
explanation = shap.Explanation(values=shap_values_single, base_values=explainer.expected_value, data=df_test.drop(columns=['prediction_label']).iloc[0])

# Waterfall Plot 생성
shap.waterfall_plot(explanation)


# In[ ]:





# In[284]:


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

# 각 클래스에 대해 개별적으로 ROC Curve 및 AUC를 계산 및 시각화
for i in range(num_classes):
    plt.figure(figsize=(8, 6))

    # i번째 클래스에 대한 이진 분류 문제 생성
    y_true_binary = np.where(y_true == i, 1, 0)
    y_pred_proba_binary = y_pred  # y_pred_proba가 아닌 y_pred를 사용합니다.

    # ROC Curve 계산
    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba_binary)
    roc_auc = auc(fpr, tpr)

    # 그래프 그리기
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Class {i})')
    plt.legend(loc='lower right')
    plt.show()


# # Evaluation Final

# In[108]:


df_test = pd.read_csv('/Users/s-alpha/Desktop/VO2peak-prediction-main/20240409_best_tuned_xgboost_VO2max_model.csv')
df_test


# ## 02.01 Rock Ports 역산

# In[110]:


# 주어진 데이터프레임을 사용하여 HR을 예측하기 위한 함수 정의
def predict_hr(age, weight_lb, vo2):
    # 수식을 사용하여 HR 예측
    predicted_hr = ((220 - age) / 2) - (vo2 * 0.09)
    return predicted_hr

# Weight를 kg에서 lb로 변환하는 함수 정의
def kg_to_lb(weight_kg):
    return weight_kg * 2.20462

# Time을 seconds에서 minutes로 변환하는 함수 정의
def sec_to_min(time_sec):
    return time_sec / 60

# 데이터프레임에 변환된 Weight(lb)와 Time(min) 열 추가
df_test['Weight_lb'] = kg_to_lb(df_test['Weight'])
df_test['Time_min'] = sec_to_min(df_test['time'])

# HR 예측을 위한 함수 호출하여 Prediction_HR 열 추가
df_test['Prediction_HR'] = predict_hr(df_test['Age'], df_test['Weight_lb'], df_test['VO2'])



# In[112]:


df_test[['Age', 'Weight', 'VO2', 'Time_min', 'Prediction_HR']]
df_test


# ## 02.02 다른 방법 역산

# In[102]:


df_test = pd.read_csv('/Users/s-alpha/Desktop/VO2peak-prediction-main/20240409_best_tuned_xgboost_VO2max_model.csv')
df_test


# In[104]:


# 최대 심박수 (MHR) 계산
df_test['MHR'] = 220 - df_test['Age']

# Haskell & Fox 식을 사용한 예상 심박수 계산
df_test['HR_Haskell_Fox'] = (60 * df_test['VO2']) / df_test['MHR']

# Karvonen 방법을 사용한 예상 심박수 계산 (휴식 심박수를 사용하지 않기 때문에 기본적인 식만 사용)
# 예상 HR = (운동 심박수 - 휴식 심박수) x Intensity + 휴식 심박수
# 여기서 휴식 심박수는 주어진 데이터에 없으므로, 예상 HR은 운동 심박수와 동일합니다.

# 결과 출력
df_test[['Age', 'Weight', 'Height', 'HR', 'VO2', 'MHR', 'HR_Haskell_Fox']]


# In[151]:


test_data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 02.00 실제 데이터 적용

# In[306]:


get_ipython().system(' ls')


# In[302]:


get_ipython().system(' pwd')


# In[318]:


real_df = pd.read_excel("/Users/s-alpha/Desktop/VO2peak-prediction-main/008SaMD.xlsx")
real_df


# In[320]:


real_df.to_csv("/Users/s-alpha/Desktop/VO2peak-prediction-main/real_data.csv", index=False)


# In[395]:


real_df = pd.read_csv("/Users/s-alpha/Desktop/VO2peak-prediction-main/real_data.csv")
real_df.head()


# In[403]:


import pandas as pd
import json



# heartRate 정보 추출 및 정렬 함수
def extract_and_sort_heart_rate(details):
    try:
        if details:
            detail_list = json.loads(details)
            sorted_heart_rates = sorted(detail_list, key=lambda x: x['time'])
            heart_rates = [item['heartRate'] for item in sorted_heart_rates if 'heartRate' in item]
            return heart_rates
        else:
            return []
    except json.JSONDecodeError:
        return []

# heartRate 정보 추출하여 새로운 열 추가
real_df['sortedHeartRates'] = real_df['details'].apply(extract_and_sort_heart_rate)

# 결과 출력
real_df


# In[405]:


real_df.to_csv("/Users/s-alpha/Desktop/VO2peak-prediction-main/real_data2.csv", index = False)


# In[459]:


real_df = pd.read_csv("/Users/s-alpha/Desktop/VO2peak-prediction-main/real_data2.csv")


# In[461]:


# 홍대언, 박세찬 등장 횟수 계산
patients_counts = real_df['patients_id'].value_counts()

# patients_id_test 열 생성
real_df['patients_id_test'] = real_df['patients_id']

for idx, row in real_df.iterrows():
    patient_id = row['patients_id']
    if patients_counts[patient_id] > 1:
        real_df.at[idx, 'patients_id_test'] = f'{patient_id}_{patients_counts[patient_id]}'
        patients_counts[patient_id] -= 1

real_df


# In[463]:


real_df


# In[465]:


# 새로운 데이터프레임 생성
result_df = pd.DataFrame(columns=['patients_id_test', 'gender_n', 'heights', 'weights', 'age', 'heartRate'])

# 리스트로 변환하고 각 값을 새로운 행으로 추가
for index, row in real_df.iterrows():
    heart_rates = row['sortedHeartRates'].strip('[]').split(', ')
    heart_rates = [int(rate) for rate in heart_rates if rate.strip()]
    heart_rate_df = pd.DataFrame({'heartRate': heart_rates})
    heart_rate_df['patients_id_test'] = row['patients_id_test']
    heart_rate_df['gender_n'] = row['gender_n']
    heart_rate_df['heights'] = row['heights']
    heart_rate_df['weights'] = row['weights']
    heart_rate_df['age'] = row['age']
    result_df = pd.concat([result_df, heart_rate_df], ignore_index=True)

# 결과 출력
result_df


# In[467]:


result_df.to_csv("/Users/s-alpha/Desktop/VO2peak-prediction-main/20240404_real_data.csv", index = False)


# ## 02.01 최종 결과

# In[257]:


final_df = pd.read_csv("/Users/s-alpha/Desktop/VO2peak-prediction-main/20240404_real_data.csv")
final_df


# In[259]:


final_df.columns


# In[261]:


# 열 이름 변경
final_df.rename(columns={'age': 'Age', 'weights': 'Weight', 
                         'heights': 'Height', 'heartRate': 'HR', 'gender_n': 'Sex'}, inplace=True)

# 결과 출력
final_df.columns


# In[263]:


final_df


# ### 02.03 진짜 VO2 값

# VO2 (ml/kg/min)= 
# 1000
# HR (bpm)×Weight (kg)×Constant (K)
# ​
# 

# In[267]:


# 상수 설정
constant_K = 3.5

# VO2 계산
final_df['VO2_ml_kg_min'] = (final_df['HR'] * final_df['Weight'] * constant_K) / 1000

# 결과 출력
final_df.head()


# In[269]:


# VO2max 범주 설정
final_df['VO2max_category'] = pd.cut(final_df['VO2_ml_kg_min'], 
                                     bins=[0, 30, 45, float('inf')], 
                                     labels=['Low', 'Moderate', 'High'])

final_df


# In[271]:


# VO2_ml_kg_min을 반올림하여 업데이트
final_df['VO2_ml_kg_min'] = final_df['VO2_ml_kg_min'].apply(np.ceil)
final_df


# In[273]:


final_df['VO2'] = final_df['Weight'] * final_df['VO2_ml_kg_min']
final_df


# In[277]:


final_df['Sex'] = final_df['Sex'].replace(1, 0)
final_df


# In[279]:


final_test_predictions = predict_model(tuned_model, data=final_df)
final_test_predictions


# In[281]:


final_df_test = final_test_predictions

# 'VO2_ml_kg_min' 및 'prediction_label' 열의 값 반올림
final_df_test['VO2_ml_kg_min'] = final_df_test['VO2_ml_kg_min'].round()
final_df_test['prediction_label'] = final_df_test['prediction_label'].round()
final_df_test


# In[283]:


final_df_test.to_csv('/Users/s-alpha/Desktop/VO2peak-prediction-main/20240404_best_tuned_xgboost_VO2max_model_result.csv', 
               index = False)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


# df_test의 'VO2_ml_kg_min' 열과 'prediction_label' 열을 각각 y_true와 y_pred로 설정
y_true = final_df_test['VO2_ml_kg_min']
y_pred = final_df_test['prediction_label']

# 예측값과 실제값 비교하여 정확도 계산
accuracy = accuracy_score(y_true, y_pred)

# 결과 출력
print(f'Accuracy: {accuracy:.4f}')


# In[285]:


from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
print("R-squared:", r2)


# In[287]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix 계산
cm = confusion_matrix(y_true, y_pred)

# 시각화
plt.figure(figsize=(32, 24))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', 
            xticklabels=[f'Predicted {i}' for i in range(1, cm.shape[1] + 1)],
            yticklabels=[f'Actual {i}' for i in range(1, cm.shape[0] + 1)])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[323]:


final_df_test


# In[329]:


train_data


# In[339]:


train_data


# In[343]:


type(tuned_model)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[66]:


age = 11
height = 163*2.54

cf = 50.72-0.372*age
pw = 0.79*height-60.7
print(cf)
print(pw)
weight = 49*2.2
print(weight)


# In[68]:


mw = weight
vo2_max = (pw+mw)/2*cf
print(vo2_max/weight)


# In[70]:


vo2_max_2 = (pw*cf)+6*(mw-pw)
print(vo2_max_2/weight)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 02. XAI

# ## 02.01. Shap [X]

# In[347]:


loaded_model = load_model('/Users/s-alpha/Desktop/VO2peak-prediction-main/20240404_best_tuned_xgb_VO2max_model')
loaded_model


# In[381]:


final_df_test


# ### 02.02 Residual Plot (잔차 그래프)

# In[387]:


import seaborn as sns
import matplotlib.pyplot as plt

# 잔차 계산
residuals = final_df_test['VO2_ml_kg_min'] - final_df_test['prediction_label']

# 잔차 그래프
plt.figure(figsize=(8, 6))
sns.residplot(x=final_df_test['prediction_label'], y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()


# ### 02.03. Prediction Plot (예측 그래프)
# 

# In[ ]:


final_df_test['VO2_ml_kg_min'] - final_df_test['prediction_label']


# In[393]:


# 예측 그래프
plt.figure(figsize=(8, 6))
plt.scatter(final_df_test['prediction_label'], final_df_test['VO2_ml_kg_min'], alpha=0.5)
plt.plot(final_df_test['prediction_label'], final_df_test['prediction_label'], color='red', linestyle='--')
plt.title('Prediction Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# ### 02.04. Feature Importance Plot (특성 중요도 그래프)

# In[398]:


# RandomForest 또는 XGBoost 모델에서 특성 중요도 확인
feature_importance = pd.Series(tuned_model.feature_importances_, 
                               index=final_df_test.drop('VO2_ml_kg_min', axis=1).columns)

feature_importance.nlargest(10).plot(kind='barh', figsize=(10, 8))

plt.title('Feature Importance Plot')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


# ## HR 역산

# In[443]:


final_df_test


# In[ ]:





# # 논문용

# In[3]:


df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




