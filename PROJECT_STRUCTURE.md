# 프로젝트 구조 개선 방안

현재 프로젝트를 더 깔끔하고 전문적으로 만들기 위한 구조 개선 제안서입니다.

## 🎯 개선 목표

1. **코드 모듈화**: 단일 파일을 기능별로 분리
2. **재사용성 향상**: 함수와 클래스 기반 구조
3. **유지보수성**: 명확한 디렉토리 구조
4. **재현성**: 설정 파일과 환경 관리
5. **문서화**: 각 모듈의 명확한 설명

## 📂 제안하는 프로젝트 구조

```
VO2max_Estimation/
│
├── README.md                    # 프로젝트 개요 및 사용법
├── PROJECT_STRUCTURE.md         # 이 문서
├── requirements.txt             # Python 패키지 의존성
├── .gitignore                   # Git 제외 파일 목록
├── setup.py                     # 패키지 설치 설정 (선택사항)
│
├── data/                        # 데이터 디렉토리
│   ├── raw/                     # 원본 데이터 (Git 제외)
│   │   ├── subject-info.csv
│   │   └── test_measure.csv
│   ├── processed/               # 전처리된 데이터
│   │   ├── train_data.csv
│   │   └── test_data.csv
│   └── README.md                # 데이터 설명 및 출처
│
├── src/                         # 소스 코드
│   ├── __init__.py
│   │
│   ├── data/                    # 데이터 처리 모듈
│   │   ├── __init__.py
│   │   ├── loader.py            # 데이터 로딩 함수
│   │   ├── preprocessor.py      # 전처리 함수
│   │   └── validator.py         # 데이터 검증 함수
│   │
│   ├── features/                # 특징 공학 모듈
│   │   ├── __init__.py
│   │   ├── engineering.py       # 특징 생성
│   │   └── selection.py         # 특징 선택
│   │
│   ├── models/                  # 모델 관련 모듈
│   │   ├── __init__.py
│   │   ├── trainer.py           # 모델 학습
│   │   ├── evaluator.py         # 모델 평가
│   │   └── predictor.py         # 예측 함수
│   │
│   ├── visualization/           # 시각화 모듈
│   │   ├── __init__.py
│   │   ├── plots.py             # 플롯 생성 함수
│   │   └── reports.py            # 리포트 생성
│   │
│   └── utils/                   # 유틸리티 함수
│       ├── __init__.py
│       ├── config.py            # 설정 관리
│       └── constants.py         # 상수 정의
│
├── notebooks/                   # Jupyter 노트북
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── scripts/                     # 실행 스크립트
│   ├── train_model.py           # 모델 학습 스크립트
│   ├── predict.py               # 예측 스크립트
│   └── evaluate.py              # 평가 스크립트
│
├── models/                      # 저장된 모델
│   ├── .gitkeep                 # 빈 디렉토리 유지
│   └── README.md                # 모델 설명
│
├── results/                     # 결과 파일
│   ├── figures/                 # 생성된 그래프
│   │   ├── distributions/
│   │   ├── correlations/
│   │   └── predictions/
│   ├── reports/                 # 평가 리포트
│   └── logs/                    # 로그 파일
│
├── tests/                       # 테스트 코드
│   ├── __init__.py
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   └── test_models.py
│
├── config/                      # 설정 파일
│   ├── config.yaml              # YAML 설정 파일
│   └── model_params.json        # 모델 하이퍼파라미터
│
└── docs/                        # 추가 문서
    ├── methodology.md           # 방법론 설명
    ├── api_reference.md         # API 참조
    └── CONTRIBUTING.md          # 기여 가이드
```

## 🔧 주요 개선 사항

### 1. 코드 모듈화

**현재**: 모든 코드가 `xgboost_ml.py` 하나의 파일에 있음

**개선**: 기능별로 모듈 분리

```python
# src/data/preprocessor.py 예시
def load_and_merge_data(subject_path, test_path):
    """데이터 로딩 및 병합"""
    pass

def preprocess_data(df):
    """데이터 전처리"""
    pass

# src/models/trainer.py 예시
def train_xgboost_model(X_train, y_train, params):
    """XGBoost 모델 학습"""
    pass
```

### 2. 설정 파일 분리

**config/config.yaml** 예시:
```yaml
data:
  subject_info_path: "data/raw/subject-info.csv"
  test_measure_path: "data/raw/test_measure.csv"
  train_test_split: 0.2
  random_state: 42

preprocessing:
  min_age: 19
  max_age: 80
  remove_multicollinearity: true
  multicollinearity_threshold: 0.90

model:
  name: "xgboost"
  cv_folds: 5
  optimize_metric: "MAPE"
  save_path: "models/best_xgboost_model.pkl"

features:
  numerical: ['Age', 'Weight', 'Height', 'HR', 'Sex', 'time']
  categorical: ['Sex']
  target: 'VO2_ml_kg_min'
```

### 3. 실행 스크립트 분리

**scripts/train_model.py** 예시:
```python
#!/usr/bin/env python
"""모델 학습 메인 스크립트"""

from src.data.loader import load_data
from src.data.preprocessor import preprocess_data
from src.models.trainer import train_model
from src.utils.config import load_config

def main():
    config = load_config('config/config.yaml')
    
    # 데이터 로딩
    df = load_data(config['data'])
    
    # 전처리
    df_processed = preprocess_data(df, config['preprocessing'])
    
    # 모델 학습
    model = train_model(df_processed, config['model'])
    
    print("Model training completed!")

if __name__ == "__main__":
    main()
```

### 4. 클래스 기반 구조 (선택사항)

더 객체지향적인 접근:

```python
# src/models/vo2_estimator.py
class VO2Estimator:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        
    def train(self, X_train, y_train, **kwargs):
        """모델 학습"""
        pass
    
    def predict(self, X):
        """예측"""
        pass
    
    def evaluate(self, X_test, y_test):
        """평가"""
        pass
    
    def save(self, path):
        """모델 저장"""
        pass
```

## 📝 구현 단계별 가이드

### Phase 1: 기본 구조 생성
1. 디렉토리 구조 생성
2. `requirements.txt` 작성
3. `.gitignore` 설정
4. 기본 `__init__.py` 파일 생성

### Phase 2: 데이터 모듈 분리
1. `src/data/loader.py` - 데이터 로딩 함수
2. `src/data/preprocessor.py` - 전처리 함수
3. `src/data/validator.py` - 검증 함수

### Phase 3: 특징 공학 모듈
1. `src/features/engineering.py` - VO2 변환, 카테고리 생성
2. `src/features/selection.py` - 특징 선택

### Phase 4: 모델 모듈
1. `src/models/trainer.py` - 모델 학습
2. `src/models/evaluator.py` - 평가 메트릭
3. `src/models/predictor.py` - 예측 함수

### Phase 5: 설정 및 스크립트
1. `config/config.yaml` 작성
2. `scripts/train_model.py` 작성
3. `scripts/predict.py` 작성

### Phase 6: 테스트 및 문서화
1. 단위 테스트 작성
2. API 문서 작성
3. 사용 예제 작성

## 🚀 빠른 시작 가이드

### 1. 구조 생성 스크립트

```bash
# 디렉토리 생성
mkdir -p data/{raw,processed}
mkdir -p src/{data,features,models,visualization,utils}
mkdir -p notebooks scripts models results/{figures,reports,logs}
mkdir -p tests config docs
```

### 2. requirements.txt 작성

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
pycaret>=3.0.0
matplotlib>=3.6.0
seaborn>=0.12.0
pyyaml>=6.0
tqdm>=4.64.0
```

### 3. .gitignore 작성

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Data
data/raw/
*.csv
*.pkl
*.h5

# Models
models/*.pkl
models/*.joblib

# Results
results/figures/*
results/reports/*
!results/figures/.gitkeep
!results/reports/.gitkeep

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

## 💡 추가 개선 제안

### 1. 로깅 시스템
```python
# src/utils/logger.py
import logging

def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    # 로깅 설정
    return logger
```

### 2. 데이터 버전 관리
- DVC (Data Version Control) 사용 고려
- 데이터셋 버전 추적

### 3. 실험 추적
- MLflow 또는 Weights & Biases 통합
- 실험 파라미터 및 결과 추적

### 4. CI/CD 파이프라인
- GitHub Actions 설정
- 자동 테스트 실행
- 코드 품질 검사

### 5. Docker 컨테이너화
```dockerfile
# Dockerfile 예시
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "scripts/train_model.py"]
```

## 📊 마이그레이션 체크리스트

- [ ] 디렉토리 구조 생성
- [ ] requirements.txt 작성
- [ ] .gitignore 설정
- [ ] 데이터 로딩 모듈 분리
- [ ] 전처리 모듈 분리
- [ ] 특징 공학 모듈 분리
- [ ] 모델 학습 모듈 분리
- [ ] 평가 모듈 분리
- [ ] 설정 파일 작성
- [ ] 실행 스크립트 작성
- [ ] 테스트 코드 작성
- [ ] 문서 업데이트
- [ ] README 업데이트

## 🎓 Best Practices

1. **명명 규칙**: PEP 8 준수
2. **타입 힌팅**: 함수 시그니처에 타입 명시
3. **Docstring**: 모든 함수에 문서화 문자열 추가
4. **에러 처리**: try-except 블록으로 예외 처리
5. **로깅**: print 대신 logging 사용
6. **테스트**: 각 모듈에 대한 단위 테스트 작성

---

이 구조를 따르면 프로젝트가 더욱 전문적이고 유지보수하기 쉬워집니다!

