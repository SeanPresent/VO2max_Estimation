"""
상수 정의 모듈
VO2max 카테고리 기준값 및 기타 상수 정의
"""

# VO2max 카테고리 기준값 (ACSM 기준)
MALE_VO2MAX_CRITERIA = {
    'Superior': [55.4, 54, 52.5, 48.9, 45.7, 42.1],
    'Excellent': [51.1, 48.3, 46.4, 43.4, 39.5, 36.7],
    'Good': [45.4, 44, 42.4, 39.2, 35.5, 32.3],
    'Fair': [41.7, 40.5, 38.5, 35.6, 32.3, 29.4]
}

FEMALE_VO2MAX_CRITERIA = {
    'Superior': [49.6, 47.4, 45.3, 41.1, 37.8, 36.7],
    'Excellent': [43.9, 42.4, 39.7, 36.7, 33, 30.9],
    'Good': [39.5, 37.8, 36.3, 33, 30, 28.1],
    'Fair': [36.1, 34.4, 33, 30.1, 27.5, 25.9]
}

# 나이 제한
MIN_AGE = 19
MAX_AGE = 80

# 성별 코드
SEX_MALE = 0
SEX_FEMALE = 1

# 특징 컬럼명
NUMERICAL_FEATURES = ['Age', 'Weight', 'Height', 'HR', 'Sex', 'time']
CATEGORICAL_FEATURES = ['Sex']
TARGET_COLUMN = 'VO2_ml_kg_min'

# 모델 기본 파라미터
DEFAULT_MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# 평가 메트릭
EVALUATION_METRICS = ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']

