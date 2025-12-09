#!/usr/bin/env python
"""
모델 학습 메인 스크립트
전체 파이프라인을 실행하여 VO2max 예측 모델을 학습합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import yaml
from src.data.preprocessor import preprocess_data
from pycaret.regression import setup, create_model, tune_model, save_model


def load_config(config_path: str = "config/config.yaml") -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_model(config: dict):
    """모델 학습 파이프라인"""
    
    # 1. 데이터 전처리
    print("Loading and preprocessing data...")
    df, excluded_count = preprocess_data(
        config['data']['subject_info_path'],
        config['data']['test_measure_path'],
        return_excluded_count=True
    )
    print(f"Excluded {excluded_count} rows due to age restrictions")
    print(f"Final dataset shape: {df.shape}")
    
    # 2. Subject-level train-test split
    print("\nSplitting data by subject...")
    from sklearn.model_selection import train_test_split
    
    unique_ids = df['ID_x'].unique()
    train_ids, test_ids = train_test_split(
        unique_ids,
        test_size=config['data']['train_test_split'],
        random_state=config['data']['random_state']
    )
    
    train_data = df[df['ID_x'].isin(train_ids)].copy()
    test_data = df[df['ID_x'].isin(test_ids)].copy()
    
    print(f"Train set: {len(train_data)} samples from {len(train_ids)} subjects")
    print(f"Test set: {len(test_data)} samples from {len(test_ids)} subjects")
    
    # 3. 학습에 필요한 컬럼만 선택
    feature_columns = config['features']['numerical'] + [config['features']['target']]
    train_data = train_data[feature_columns]
    
    # 4. PyCaret 설정
    print("\nSetting up PyCaret...")
    exp = setup(
        data=train_data,
        target=config['features']['target'],
        session_id=config['model']['session_id'],
        transformation=config['preprocessing']['transformation'],
        fold=config['model']['cv_folds'],
        numeric_features=config['features']['numerical'],
        remove_multicollinearity=config['preprocessing']['remove_multicollinearity'],
        multicollinearity_threshold=config['preprocessing']['multicollinearity_threshold'],
        silent=True
    )
    
    # 5. 모델 생성
    print(f"\nCreating {config['model']['name']} model...")
    model = create_model(config['model']['name'], fold=config['model']['cv_folds'])
    
    # 6. 하이퍼파라미터 튜닝
    print(f"\nTuning model (optimizing {config['model']['optimize_metric']})...")
    tuned_model = tune_model(
        model,
        fold=config['model']['cv_folds'],
        optimize=config['model']['optimize_metric'],
        choose_better=True,
        return_train_score=True
    )
    
    # 7. 모델 저장
    model_path = config['model']['save_path']
    print(f"\nSaving model to {model_path}...")
    save_model(tuned_model, model_path)
    
    print("\nModel training completed successfully!")
    return tuned_model


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train VO2max estimation model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 모델 학습
    model = train_model(config)


if __name__ == "__main__":
    main()

