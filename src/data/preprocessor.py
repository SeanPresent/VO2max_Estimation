"""
데이터 전처리 모듈
데이터 로딩, 병합, 정제, 변환 함수 제공
"""

import pandas as pd
import numpy as np
import math
from typing import Tuple, Optional
from ..utils.constants import (
    MALE_VO2MAX_CRITERIA, 
    FEMALE_VO2MAX_CRITERIA,
    MIN_AGE,
    MAX_AGE,
    SEX_MALE,
    TARGET_COLUMN
)


def load_and_merge_data(subject_path: str, test_path: str) -> pd.DataFrame:
    """
    데이터 로딩 및 병합
    
    Args:
        subject_path: subject-info.csv 파일 경로
        test_path: test_measure.csv 파일 경로
        
    Returns:
        병합된 DataFrame
    """
    sub_df = pd.read_csv(subject_path, low_memory=False)
    test_measure_df = pd.read_csv(test_path, low_memory=False)
    
    # 데이터 병합
    df = pd.merge(sub_df, test_measure_df, on='ID_test')
    
    # VO2가 없는 행 제거
    df.dropna(subset=['VO2'], inplace=True)
    
    return df


def convert_vo2_to_ml_kg_min(df: pd.DataFrame) -> pd.DataFrame:
    """
    VO2를 mL/min에서 mL/kg/min으로 변환
    
    Args:
        df: 원본 DataFrame
        
    Returns:
        변환된 DataFrame
    """
    df = df.copy()
    df[TARGET_COLUMN] = df['VO2'] / df['Weight']
    df['Age'] = df['Age'].apply(math.ceil)
    return df


def categorize_vo2max(row: pd.Series) -> Optional[str]:
    """
    VO2max 값을 카테고리로 분류
    
    Args:
        row: 데이터 행 (Age, Sex, VO2_ml_kg_min 포함)
        
    Returns:
        카테고리 문자열 또는 None
    """
    age = row['Age']
    sex = row['Sex']
    vo2_ml_kg_min = row[TARGET_COLUMN]
    
    # 나이 제한 확인
    if age <= MIN_AGE or age >= MAX_AGE:
        return None
    
    # 성별에 따른 기준 선택
    criteria = MALE_VO2MAX_CRITERIA if sex == SEX_MALE else FEMALE_VO2MAX_CRITERIA
    
    # 카테고리 분류
    for category, values in criteria.items():
        if vo2_ml_kg_min >= values[0]:
            return category
    
    return "Poor"


def add_vo2max_category(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    VO2max 카테고리 컬럼 추가
    
    Args:
        df: DataFrame
        
    Returns:
        (카테고리가 추가된 DataFrame, 제외된 행 수)
    """
    df = df.copy()
    excluded_rows = 0
    
    def vo2max_category_wrapper(row):
        nonlocal excluded_rows
        result = categorize_vo2max(row)
        if result is None:
            excluded_rows += 1
        return result
    
    df['VO2max_category'] = df.apply(vo2max_category_wrapper, axis=1)
    
    return df, excluded_rows


def filter_valid_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    유효한 카테고리만 필터링
    
    Args:
        df: DataFrame
        
    Returns:
        필터링된 DataFrame
    """
    valid_categories = ['Poor', 'Fair', 'Good', 'Excellent', 'Superior']
    filtered_df = df[df['VO2max_category'].isin(valid_categories)].copy()
    return filtered_df


def clean_vo2_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    VO2 데이터 정제
    
    Args:
        df: DataFrame
        
    Returns:
        정제된 DataFrame
    """
    df = df.copy()
    
    # 음수 값 제거
    df = df[df[TARGET_COLUMN] >= 0]
    
    # 결측값 제거
    df = df.dropna(subset=[TARGET_COLUMN])
    
    # 소수점 반올림
    df[TARGET_COLUMN] = df[TARGET_COLUMN].round(1)
    
    return df


def preprocess_data(
    subject_path: str,
    test_path: str,
    return_excluded_count: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, int]:
    """
    전체 전처리 파이프라인
    
    Args:
        subject_path: subject-info.csv 파일 경로
        test_path: test_measure.csv 파일 경로
        return_excluded_count: 제외된 행 수 반환 여부
        
    Returns:
        전처리된 DataFrame (및 제외된 행 수)
    """
    # 1. 데이터 로딩 및 병합
    df = load_and_merge_data(subject_path, test_path)
    
    # 2. VO2 단위 변환
    df = convert_vo2_to_ml_kg_min(df)
    
    # 3. VO2max 카테고리 추가
    df, excluded_count = add_vo2max_category(df)
    
    # 4. 유효한 카테고리만 필터링
    df = filter_valid_categories(df)
    
    # 5. 데이터 정제
    df = clean_vo2_data(df)
    
    if return_excluded_count:
        return df, excluded_count
    return df

