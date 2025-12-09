# 빠른 시작 가이드

이 가이드는 프로젝트를 빠르게 시작하는 방법을 설명합니다.

## 📋 사전 요구사항

- Python 3.8 이상
- pip 패키지 관리자
- Git (선택사항)

## 🚀 설치 및 실행

### 1단계: 프로젝트 구조 생성

```bash
# 프로젝트 디렉토리로 이동
cd VO2max_Estimation

# 프로젝트 구조 생성 스크립트 실행
bash setup_project.sh
```

또는 수동으로 디렉토리를 생성할 수 있습니다.

### 2단계: 가상 환경 생성 및 활성화

```bash
# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3단계: 패키지 설치

```bash
pip install -r requirements.txt
```

### 4단계: 데이터 준비

```bash
# 데이터 디렉토리에 원본 데이터 파일 배치
# data/raw/subject-info.csv
# data/raw/test_measure.csv
```

### 5단계: 설정 파일 생성

```bash
# 설정 파일 예시 복사
cp config/config.yaml.example config/config.yaml

# 필요에 따라 config.yaml 수정
```

### 6단계: 모델 학습

```bash
# 기본 설정으로 학습
python scripts/train_model.py

# 또는 커스텀 설정 파일 사용
python scripts/train_model.py --config config/my_config.yaml
```

## 📝 기본 사용 예제

### Python에서 직접 사용

```python
from src.data.preprocessor import preprocess_data
from pycaret.regression import load_model, predict_model
import pandas as pd

# 1. 데이터 전처리
df = preprocess_data(
    subject_path="data/raw/subject-info.csv",
    test_path="data/raw/test_measure.csv"
)

# 2. 모델 로드
model = load_model("models/best_xgboost_model.pkl")

# 3. 예측
new_data = pd.DataFrame({
    'Age': [30],
    'Weight': [70],
    'Height': [175],
    'HR': [150],
    'Sex': [0],
    'time': [10]
})

predictions = predict_model(model, data=new_data)
print(f"Predicted VO2max: {predictions['Label'].values[0]} mL/kg/min")
```

### Jupyter Notebook 사용

```python
# notebooks/01_data_exploration.ipynb에서 시작
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from src.data.preprocessor import preprocess_data
import pandas as pd

# 데이터 로딩 및 탐색
df = preprocess_data("data/raw/subject-info.csv", "data/raw/test_measure.csv")
df.head()
df.describe()
```

## 🔍 프로젝트 구조 이해하기

```
VO2max_Estimation/
├── data/           # 데이터 파일
├── src/            # 소스 코드 모듈
├── scripts/        # 실행 스크립트
├── notebooks/      # Jupyter 노트북
├── models/         # 저장된 모델
├── results/        # 결과 파일
└── config/         # 설정 파일
```

## 📚 다음 단계

1. **데이터 탐색**: `notebooks/01_data_exploration.ipynb` 실행
2. **특징 공학**: `notebooks/02_feature_engineering.ipynb` 실행
3. **모델 학습**: `scripts/train_model.py` 실행
4. **모델 평가**: `notebooks/04_model_evaluation.ipynb` 실행

## ❓ 문제 해결

### 일반적인 문제

1. **모듈을 찾을 수 없음**
   ```bash
   # 프로젝트 루트에서 실행하는지 확인
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **데이터 파일을 찾을 수 없음**
   - `config/config.yaml`에서 데이터 경로 확인
   - 파일이 `data/raw/` 디렉토리에 있는지 확인

3. **PyCaret 설치 오류**
   ```bash
   pip install --upgrade pycaret
   ```

## 📖 추가 문서

- [README.md](README.md) - 프로젝트 개요
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 프로젝트 구조 상세 설명
- [docs/methodology.md](docs/methodology.md) - 방법론 설명 (작성 예정)

## 💡 팁

- 개발 중에는 Jupyter Notebook을 사용하는 것이 편리합니다
- 프로덕션 환경에서는 `scripts/`의 스크립트를 사용하세요
- 설정 파일을 통해 하이퍼파라미터를 쉽게 조정할 수 있습니다

