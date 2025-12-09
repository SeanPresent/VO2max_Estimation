#!/bin/bash

# VO2max Estimation 프로젝트 구조 생성 스크립트

echo "Creating project structure..."

# 디렉토리 생성
mkdir -p data/raw
mkdir -p data/processed
mkdir -p src/data
mkdir -p src/features
mkdir -p src/models
mkdir -p src/visualization
mkdir -p src/utils
mkdir -p notebooks
mkdir -p scripts
mkdir -p models
mkdir -p results/figures
mkdir -p results/reports
mkdir -p results/logs
mkdir -p tests
mkdir -p config
mkdir -p docs

# .gitkeep 파일 생성 (빈 디렉토리 유지)
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/.gitkeep
touch results/figures/.gitkeep
touch results/reports/.gitkeep
touch results/logs/.gitkeep

# __init__.py 파일 생성
touch src/__init__.py
touch src/data/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
touch src/visualization/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

echo "Project structure created successfully!"
echo ""
echo "Next steps:"
echo "1. Place your data files in data/raw/"
echo "2. Start migrating code from xgboost_ml.py to the new structure"
echo "3. Create config/config.yaml with your settings"
echo "4. Run: pip install -r requirements.txt"

