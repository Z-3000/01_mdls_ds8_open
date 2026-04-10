#!/usr/bin/env python3
"""
House Price Prediction - Rev.12 Fast (SSH 실행용)
Usage: python3 run_rev12_fast.py
예상 소요: i7-6700HQ 기준 30~40분
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, LassoCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                               ExtraTreesRegressor, HistGradientBoostingRegressor)
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from scipy.optimize import minimize
import shutil

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
CPU_CORES = os.cpu_count() or 4

# GPU 자동 감지
USE_GPU = False
try:
    _test = xgb.XGBRegressor(tree_method='hist', device='cuda:0', n_estimators=1)
    _test.fit(np.array([[1,2],[3,4]]), np.array([1,2]))
    USE_GPU = True
    print("[INFO] GPU (CUDA) 사용 가능 ✓")
except Exception:
    print("[INFO] GPU 없음, CPU 모드로 실행")

print(f"[INFO] CPU: {CPU_CORES} cores, GPU: {USE_GPU}")
print(f"[INFO] 작업 디렉토리: {os.getcwd()}")

T_GLOBAL = time.time()

# ===================================================================
# 데이터 로드
# ===================================================================
print("\n" + "=" * 60)
print("[1/6] 데이터 로드")
print("=" * 60)

# 데이터 파일 경로 탐색
script_dir = os.path.dirname(os.path.abspath(__file__))
for try_dir in [script_dir, os.getcwd(), '.']:
    train_path = os.path.join(try_dir, 'train.csv')
    if os.path.exists(train_path):
        break
else:
    print("[ERROR] train.csv / test.csv 파일을 찾을 수 없습니다!")
    print(f"  현재 디렉토리: {os.getcwd()}")
    print(f"  스크립트 디렉토리: {script_dir}")
    sys.exit(1)

train_df = pd.read_csv(os.path.join(try_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(try_dir, 'test.csv'))
print(f"  Train: {train_df.shape}, Test: {test_df.shape}")

# ===================================================================
# 전처리 파이프라인
# ===================================================================
print("\n" + "=" * 60)
print("[2/6] 전처리 + 피처 엔지니어링")
print("=" * 60)
t0 = time.time()

train_len = len(train_df)
test_ids = test_df['Id'].copy()
y_train_full = np.log1p(train_df['SalePrice'])

X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
X_test = test_df.drop(['Id'], axis=1)
all_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

# 결측치 처리
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'MasVnrType']
for col in none_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')

zero_cols = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
             'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']
for col in zero_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(0)

if 'LotFrontage' in all_data.columns and 'Neighborhood' in all_data.columns:
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))

for col in all_data.select_dtypes(include=[np.number]).columns:
    if all_data[col].isnull().sum() > 0:
        all_data[col] = all_data[col].fillna(all_data[col].median())

for col in all_data.select_dtypes(include=['object']).columns:
    if all_data[col].isnull().sum() > 0:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# 희귀 카테고리 통합
for col in all_data.select_dtypes(include=['object']).columns:
    freq = all_data[col].value_counts(normalize=True)
    rare = freq[freq < 0.01].index
    if len(rare) > 0:
        all_data[col] = all_data[col].replace(rare, 'Rare')

# 피처 엔지니어링
def create_features(df):
    df = df.copy()
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    df['TotalBath'] = df['FullBath'] + df['HalfBath'] * 0.5 + df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5
    df['TotalLivingArea'] = df['GrLivArea'] + df['TotalBsmtSF']
    df['TotalHouseArea'] = df['TotalSF'] + df['GarageArea'] + df['TotalPorchSF']
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
    df['IsNewHouse'] = (df['YrSold'] == df['YearBuilt']).astype(int)
    df['MoSold_sin'] = np.sin(2 * np.pi * df['MoSold'] / 12)
    df['MoSold_cos'] = np.cos(2 * np.pi * df['MoSold'] / 12)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['OverallQual_x_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
    df['OverallQual_x_TotalSF'] = df['OverallQual'] * df['TotalSF']
    df['OverallQual_x_TotalBsmtSF'] = df['OverallQual'] * df['TotalBsmtSF']
    df['OverallQual_x_GarageArea'] = df['OverallQual'] * df['GarageArea']
    df['GrLivArea_x_TotalBsmtSF'] = df['GrLivArea'] * df['TotalBsmtSF']
    df['LotArea_x_OverallQual'] = df['LotArea'] * df['OverallQual']
    df['OverallQual_sq'] = df['OverallQual'] ** 2
    df['GrLivArea_sq'] = df['GrLivArea'] ** 2
    df['TotalSF_sq'] = df['TotalSF'] ** 2
    df['OverallQual_cu'] = df['OverallQual'] ** 3
    df['BsmtFinRatio'] = np.where(df['TotalBsmtSF'] > 0, df['BsmtFinSF1'] / df['TotalBsmtSF'], 0)
    df['LivAreaRatio'] = df['GrLivArea'] / df['LotArea'].clip(lower=1)
    df['GaragePerCar'] = np.where(df['GarageCars'] > 0, df['GarageArea'] / df['GarageCars'], 0)
    df['BathsPerRoom'] = df['TotalBath'] / df['TotRmsAbvGrd'].clip(lower=1)
    df['GarageAge'] = np.where(df['GarageYrBlt'] > 0, df['YrSold'] - df['GarageYrBlt'], 0)
    df['TotalAge'] = df['HouseAge'] + df['RemodAge']
    df['Age_x_Qual'] = df['HouseAge'] * df['OverallQual']
    df['RemodAge_x_Qual'] = df['RemodAge'] * df['OverallQual']
    return df

all_data = create_features(all_data)

# 순서형 인코딩
quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0, 'NA': 0}
for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
            'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
    if col in all_data.columns:
        all_data[col] = all_data[col].map(quality_map).fillna(0)

for col, mapping in [
    ('BsmtExposure', {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}),
    ('GarageFinish', {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0}),
    ('Fence', {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'None': 0}),
    ('Functional', {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0}),
    ('PavedDrive', {'Y': 2, 'P': 1, 'N': 0}),
    ('CentralAir', {'Y': 1, 'N': 0}),
    ('Street', {'Pave': 1, 'Grvl': 0}),
]:
    if col in all_data.columns:
        default = 7 if col == 'Functional' else 0
        all_data[col] = all_data[col].map(mapping).fillna(default)

for col in ['BsmtFinType1', 'BsmtFinType2']:
    if col in all_data.columns:
        all_data[col] = all_data[col].map(
            {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}).fillna(0)

# 품질 종합 점수
all_data['TotalQualScore'] = (all_data['OverallQual'] + all_data['OverallCond'] +
    all_data['ExterQual'] + all_data['ExterCond'] + all_data['BsmtQual'] +
    all_data['KitchenQual'] + all_data['GarageQual'] + all_data['HeatingQC'])
all_data['QualCondProduct'] = all_data['OverallQual'] * all_data['OverallCond']
all_data['ExterQual_x_GrLivArea'] = all_data['ExterQual'] * all_data['GrLivArea']
all_data['KitchenQual_x_GrLivArea'] = all_data['KitchenQual'] * all_data['GrLivArea']
all_data['BsmtQual_x_TotalBsmtSF'] = all_data['BsmtQual'] * all_data['TotalBsmtSF']
all_data['TotalQualScore_x_TotalSF'] = all_data['TotalQualScore'] * all_data['TotalSF']
all_data['TotalQualScore_x_GrLivArea'] = all_data['TotalQualScore'] * all_data['GrLivArea']
all_data['OverallQual_x_TotalBath'] = all_data['OverallQual'] * all_data['TotalBath']
all_data['BsmtToTotal'] = np.where(all_data['TotalSF'] > 0, all_data['TotalBsmtSF'] / all_data['TotalSF'], 0)
all_data['GarageToLiving'] = np.where(all_data['GrLivArea'] > 0, all_data['GarageArea'] / all_data['GrLivArea'], 0)

# K-Fold Target Encoding
print("  Target Encoding...")
kf_te = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
te_columns = ['Neighborhood', 'Exterior1st', 'Exterior2nd', 'MSSubClass']
train_part = all_data.iloc[:train_len]
test_part = all_data.iloc[train_len:]

for col in te_columns:
    if col not in all_data.columns:
        continue
    train_encoded = pd.Series(np.zeros(train_len), index=train_part.index)
    for tr_idx, val_idx in kf_te.split(train_part):
        fold_train = train_part.iloc[tr_idx]
        fold_target = y_train_full.iloc[tr_idx]
        global_mean = fold_target.mean()
        agg = fold_target.groupby(fold_train[col]).agg(['mean', 'count'])
        smoothed = (agg['count'] * agg['mean'] + 10 * global_mean) / (agg['count'] + 10)
        train_encoded.iloc[val_idx] = train_part.iloc[val_idx][col].map(smoothed).fillna(global_mean)

    global_mean = y_train_full.mean()
    agg = y_train_full.groupby(train_part[col]).agg(['mean', 'count'])
    smoothed = (agg['count'] * agg['mean'] + 10 * global_mean) / (agg['count'] + 10)
    test_encoded = test_part[col].map(smoothed).fillna(global_mean)

    all_data.loc[train_part.index, f'{col}_te'] = train_encoded.values
    all_data.loc[test_part.index, f'{col}_te'] = test_encoded.values

# TE 상호작용 피처 (5.5단계)
for te_col in ['Neighborhood_te', 'Exterior1st_te', 'Exterior2nd_te', 'MSSubClass_te']:
    if te_col in all_data.columns:
        all_data[f'{te_col}_x_OverallQual'] = all_data[te_col] * all_data['OverallQual']
        all_data[f'{te_col}_x_GrLivArea'] = all_data[te_col] * all_data['GrLivArea']
        if te_col == 'Neighborhood_te':
            all_data[f'{te_col}_x_TotalSF'] = all_data[te_col] * all_data['TotalSF']

# 명목형 인코딩
cat_cols = all_data.select_dtypes(include=['object']).columns.tolist()
onehot_cols = [c for c in cat_cols if all_data[c].nunique() <= 10]
drop_cols = [c for c in cat_cols if c not in onehot_cols]

for col in drop_cols:
    te_col = f'{col}_te'
    if te_col not in all_data.columns:
        freq_map = all_data[col].value_counts(normalize=True).to_dict()
        all_data[col + '_freq'] = all_data[col].map(freq_map)

if onehot_cols:
    all_data = pd.get_dummies(all_data, columns=onehot_cols, drop_first=True, dtype=int)
all_data = all_data.drop([c for c in drop_cols if c in all_data.columns], axis=1)
remaining_obj = all_data.select_dtypes(include=['object']).columns
if len(remaining_obj) > 0:
    all_data = all_data.drop(remaining_obj, axis=1)

# 이상치 제거 (BoxCox 전!)
_train_part = all_data.iloc[:train_len]
_outlier_mask = (_train_part['GrLivArea'] > 4000) & (y_train_full < np.log1p(300000))
_outlier_idx = _train_part[_outlier_mask].index.tolist()
if len(_outlier_idx) > 0:
    all_data = all_data.drop(_outlier_idx).reset_index(drop=True)
    y_train_full = y_train_full.drop(_outlier_idx).reset_index(drop=True)
    train_len -= len(_outlier_idx)
    print(f"  이상치 {len(_outlier_idx)}개 제거")

# 왜도 처리 (BoxCox)
numeric_feats = all_data.select_dtypes(include=[np.number]).columns
for feat in numeric_feats:
    if abs(all_data[feat].skew()) > 0.75:
        try:
            if all_data[feat].std() < 1e-10:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                lam = boxcox_normmax(all_data[feat] + 1)
                all_data[feat] = boxcox1p(all_data[feat], lam)
        except Exception:
            if all_data[feat].min() >= 0:
                all_data[feat] = np.log1p(all_data[feat])

# 수치 안정화
num_cols = all_data.select_dtypes(include=[np.number]).columns
all_data[num_cols] = all_data[num_cols].replace([np.inf, -np.inf], np.nan)
for col in num_cols:
    if all_data[col].isnull().any():
        all_data[col] = all_data[col].fillna(all_data[col].median())
    q_lo, q_hi = all_data[col].quantile([0.001, 0.999])
    if np.isfinite(q_lo) and np.isfinite(q_hi) and q_lo < q_hi:
        all_data[col] = all_data[col].clip(q_lo, q_hi)

# 데이터 분리
X_train_raw = all_data.iloc[:train_len].astype(np.float64)
X_sub_raw = all_data.iloc[train_len:].astype(np.float64)
y_train_clean = y_train_full.copy()

for col in X_train_raw.columns:
    if X_train_raw[col].isnull().sum() > 0:
        med = X_train_raw[col].median()
        X_train_raw[col] = X_train_raw[col].fillna(med)
        X_sub_raw[col] = X_sub_raw[col].fillna(med)

X_train_raw = X_train_raw.replace([np.inf, -np.inf], 0)
X_sub_raw = X_sub_raw.replace([np.inf, -np.inf], 0)

# 스케일링
scaler = RobustScaler()
feature_names = X_train_raw.columns.tolist()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feature_names).astype(np.float64)
X_sub_scaled = pd.DataFrame(scaler.transform(X_sub_raw), columns=feature_names).astype(np.float64)

print(f"  완료: {X_train_scaled.shape[1]}개 피처, {time.time()-t0:.1f}초")

# ===================================================================
# 피처 선택
# ===================================================================
print("\n" + "=" * 60)
print("[3/6] Lasso 피처 선택 + Optuna 튜닝")
print("=" * 60)
t0 = time.time()

lasso_cv = LassoCV(alphas=np.logspace(-6, -2, 100), cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_scaled, y_train_clean)

selected = pd.Series(np.abs(lasso_cv.coef_), index=X_train_scaled.columns)
selected = selected[selected > 0].index.tolist()
print(f"  {len(X_train_scaled.columns)} → {len(selected)}개 피처 선택")

X_tr_s = X_train_scaled[selected]
X_sub_s = X_sub_scaled[selected]
X_tr_r = X_train_raw[selected]
X_sub_r = X_sub_raw[selected]

# ===================================================================
# Optuna 튜닝 (속도 최적화: trials 축소)
# ===================================================================
kf_opt = KFold(n_splits=5, shuffle=True, random_state=42)

xgb_extra = dict(random_state=42, tree_method='hist', n_jobs=CPU_CORES, verbosity=0)
lgb_extra = dict(random_state=42, n_jobs=CPU_CORES, verbosity=-1)
if USE_GPU:
    xgb_extra['device'] = 'cuda:0'
    lgb_extra['device'] = 'gpu'

# ElasticNet (30 trials)
def obj_enet(trial):
    a = trial.suggest_float('alpha', 1e-6, 0.01, log=True)
    l = trial.suggest_float('l1_ratio', 0.01, 0.99)
    m = ElasticNet(alpha=a, l1_ratio=l, random_state=42, max_iter=10000)
    return -cross_val_score(m, X_tr_s, y_train_clean, scoring='neg_root_mean_squared_error', cv=kf_opt, n_jobs=1).mean()
st_enet = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
st_enet.optimize(obj_enet, n_trials=30, show_progress_bar=False)
print(f"  ElasticNet: {st_enet.best_value:.5f} ({time.time()-t0:.0f}s)")

# KernelRidge (30 trials)
def obj_kr(trial):
    a = trial.suggest_float('alpha', 0.01, 50.0, log=True)
    d = trial.suggest_int('degree', 2, 3)
    c = trial.suggest_float('coef0', 0.1, 10.0)
    m = KernelRidge(alpha=a, kernel='polynomial', degree=d, coef0=c)
    return -cross_val_score(m, X_tr_s, y_train_clean, scoring='neg_root_mean_squared_error', cv=kf_opt, n_jobs=1).mean()
st_kr = optuna.create_study(direction='minimize', sampler=TPESampler(seed=123))
st_kr.optimize(obj_kr, n_trials=30, show_progress_bar=False)
print(f"  KernelRidge: {st_kr.best_value:.5f} ({time.time()-t0:.0f}s)")

# GradientBoosting (30 trials, max 1500 estimators)
def obj_gb(trial):
    p = {'n_estimators': trial.suggest_int('n_estimators', 200, 1500, step=100),
         'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
         'max_depth': trial.suggest_int('max_depth', 3, 6),
         'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
         'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
         'max_features': trial.suggest_float('max_features', 0.3, 1.0)}
    m = GradientBoostingRegressor(**p, random_state=42)
    return -cross_val_score(m, X_tr_r, y_train_clean, scoring='neg_root_mean_squared_error', cv=kf_opt, n_jobs=1).mean()
st_gb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=456))
st_gb.optimize(obj_gb, n_trials=30, show_progress_bar=False)
print(f"  GradientBoosting: {st_gb.best_value:.5f} ({time.time()-t0:.0f}s)")

# XGBoost (40 trials, max 3000 estimators)
def obj_xgb(trial):
    p = {'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=100),
         'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
         'max_depth': trial.suggest_int('max_depth', 3, 7),
         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
         'gamma': trial.suggest_float('gamma', 0.0, 3.0),
         'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
         'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True)}
    m = xgb.XGBRegressor(**p, **xgb_extra)
    return -cross_val_score(m, X_tr_r, y_train_clean, scoring='neg_root_mean_squared_error', cv=kf_opt, n_jobs=1).mean()
st_xgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=789))
st_xgb.optimize(obj_xgb, n_trials=40, show_progress_bar=False)
print(f"  XGBoost: {st_xgb.best_value:.5f} ({time.time()-t0:.0f}s)")

# LightGBM (40 trials)
def obj_lgb(trial):
    p = {'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=100),
         'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
         'max_depth': trial.suggest_int('max_depth', 3, 8),
         'num_leaves': trial.suggest_int('num_leaves', 15, 100),
         'min_child_samples': trial.suggest_int('min_child_samples', 5, 40),
         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
         'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
         'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True)}
    m = lgb.LGBMRegressor(**p, **lgb_extra)
    return -cross_val_score(m, X_tr_r, y_train_clean, scoring='neg_root_mean_squared_error', cv=kf_opt, n_jobs=1).mean()
st_lgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=2024))
st_lgb.optimize(obj_lgb, n_trials=40, show_progress_bar=False)
print(f"  LightGBM: {st_lgb.best_value:.5f} ({time.time()-t0:.0f}s)")

print(f"\n  Optuna 완료: {(time.time()-t0)/60:.1f}분")

# ===================================================================
# 모델 설정
# ===================================================================
SEEDS = [42, 123, 456]  # 3개 (속도 최적화)

model_configs = {
    'ElasticNet':       {'params': dict(**st_enet.best_params, random_state=42, max_iter=10000),
                         'cls': ElasticNet, 'data': 'scaled', 'seeds': None, 'group': 'stack'},
    'KernelRidge':      {'params': dict(**st_kr.best_params, kernel='polynomial'),
                         'cls': KernelRidge, 'data': 'scaled', 'seeds': None, 'group': 'stack'},
    'GradientBoosting': {'params': dict(**st_gb.best_params, random_state=42),
                         'cls': GradientBoostingRegressor, 'data': 'raw', 'seeds': SEEDS, 'group': 'stack'},
    'XGBoost':          {'params': dict(**st_xgb.best_params, **xgb_extra),
                         'cls': xgb.XGBRegressor, 'data': 'raw', 'seeds': SEEDS, 'group': 'boost'},
    'LightGBM':         {'params': dict(**st_lgb.best_params, **lgb_extra),
                         'cls': lgb.LGBMRegressor, 'data': 'raw', 'seeds': SEEDS, 'group': 'boost'},
}

model_names = list(model_configs.keys())
stack_names = [n for n, c in model_configs.items() if c['group'] == 'stack']
boost_names = [n for n, c in model_configs.items() if c['group'] == 'boost']

# ===================================================================
# Custom OOF Stacking
# ===================================================================
print("\n" + "=" * 60)
print("[4/6] Custom OOF Stacking (5-fold)")
print("=" * 60)
t0 = time.time()

N_FOLDS = 5
kf_oof = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = {}
test_preds_oof = {}
oof_scores = {}

for name, cfg in model_configs.items():
    X_use = X_tr_s if cfg['data'] == 'scaled' else X_tr_r
    X_test_use = X_sub_s if cfg['data'] == 'scaled' else X_sub_r

    oof_pred = np.zeros(len(y_train_clean))
    test_fold_preds = np.zeros((len(X_test_use), N_FOLDS))

    for fold_i, (tr_idx, val_idx) in enumerate(kf_oof.split(X_use)):
        Xf_tr, Xf_val = X_use.iloc[tr_idx], X_use.iloc[val_idx]
        yf_tr = y_train_clean.iloc[tr_idx]

        if cfg['seeds'] is not None:
            val_seeds, test_seeds = [], []
            for seed in cfg['seeds']:
                params = cfg['params'].copy()
                params['random_state'] = seed
                m = cfg['cls'](**params)
                m.fit(Xf_tr, yf_tr)
                val_seeds.append(m.predict(Xf_val))
                test_seeds.append(m.predict(X_test_use))
            oof_pred[val_idx] = np.mean(val_seeds, axis=0)
            test_fold_preds[:, fold_i] = np.mean(test_seeds, axis=0)
        else:
            m = cfg['cls'](**cfg['params'])
            m.fit(Xf_tr, yf_tr)
            oof_pred[val_idx] = m.predict(Xf_val)
            test_fold_preds[:, fold_i] = m.predict(X_test_use)

    oof_preds[name] = oof_pred
    test_preds_oof[name] = test_fold_preds.mean(axis=1)
    score = np.sqrt(mean_squared_error(y_train_clean, oof_pred))
    oof_scores[name] = score
    print(f"  {name} OOF RMSE: {score:.5f} ({time.time()-t0:.0f}s)")

# Stacking: Stack 그룹 → Lasso 메타
print("\n  Stacking: Stack 그룹 → Lasso 메타 모델")
oof_stack = np.column_stack([oof_preds[n] for n in stack_names])
test_stack = np.column_stack([test_preds_oof[n] for n in stack_names])

meta_lasso = LassoCV(alphas=np.logspace(-6, -2, 100), cv=5, max_iter=10000, random_state=42)
meta_lasso.fit(oof_stack, y_train_clean)

stacked_oof = meta_lasso.predict(oof_stack)
stacked_test = meta_lasso.predict(test_stack)
stacked_rmse = np.sqrt(mean_squared_error(y_train_clean, stacked_oof))
print(f"  Stacked OOF RMSE: {stacked_rmse:.5f}")
print(f"  Meta 가중치: {dict(zip(stack_names, meta_lasso.coef_.round(4)))}")

# ===================================================================
# 블렌딩 비율 탐색
# ===================================================================
print("\n" + "=" * 60)
print("[5/6] 블렌딩 비율 탐색")
print("=" * 60)

xgb_oof = oof_preds['XGBoost']
lgb_oof = oof_preds['LightGBM']
xgb_test = test_preds_oof['XGBoost']
lgb_test = test_preds_oof['LightGBM']

best_rmse = float('inf')
best_w = (0.70, 0.15, 0.15)

for ws in np.arange(0.50, 0.85, 0.05):
    for wx in np.arange(0.05, 0.30, 0.05):
        wl = 1.0 - ws - wx
        if wl < 0.05 or wl > 0.30:
            continue
        blend = ws * stacked_oof + wx * xgb_oof + wl * lgb_oof
        rmse = np.sqrt(mean_squared_error(y_train_clean, blend))
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = (round(ws, 2), round(wx, 2), round(wl, 2))

serigne_blend = 0.70 * stacked_oof + 0.15 * xgb_oof + 0.15 * lgb_oof
serigne_rmse = np.sqrt(mean_squared_error(y_train_clean, serigne_blend))

print(f"  최적 비율: Stack={best_w[0]}, XGB={best_w[1]}, LGB={best_w[2]}")
print(f"  최적 OOF RMSE: {best_rmse:.5f}")
print(f"  Serigne(70/15/15) OOF RMSE: {serigne_rmse:.5f}")

# Full-data Stack 학습
print("\n  Full-data Stack 모델 학습...")
test_stack_full = np.zeros((len(X_sub_s), len(stack_names)))
for i, name in enumerate(stack_names):
    cfg = model_configs[name]
    X_tr = X_tr_s if cfg['data'] == 'scaled' else X_tr_r
    X_sub = X_sub_s if cfg['data'] == 'scaled' else X_sub_r
    if cfg['seeds']:
        preds = []
        for seed in cfg['seeds']:
            params = cfg['params'].copy()
            params['random_state'] = seed
            m = cfg['cls'](**params)
            m.fit(X_tr, y_train_clean)
            preds.append(m.predict(X_sub))
        test_stack_full[:, i] = np.mean(preds, axis=0)
    else:
        m = cfg['cls'](**cfg['params'])
        m.fit(X_tr, y_train_clean)
        test_stack_full[:, i] = m.predict(X_sub)
    print(f"    {name} 완료")

stacked_full = meta_lasso.predict(test_stack_full)

# Full-data Boost 학습
xgb_full_preds, lgb_full_preds = [], []
for seed in SEEDS:
    p = dict(**st_xgb.best_params, **xgb_extra)
    p['random_state'] = seed
    m = xgb.XGBRegressor(**p)
    m.fit(X_tr_r, y_train_clean)
    xgb_full_preds.append(m.predict(X_sub_r))

    p2 = dict(**st_lgb.best_params, **lgb_extra)
    p2['random_state'] = seed
    m2 = lgb.LGBMRegressor(**p2)
    m2.fit(X_tr_r, y_train_clean)
    lgb_full_preds.append(m2.predict(X_sub_r))

xgb_full = np.mean(xgb_full_preds, axis=0)
lgb_full = np.mean(lgb_full_preds, axis=0)

print(f"\n  전체 학습 완료: {(time.time()-t0)/60:.1f}분")

# ===================================================================
# 제출 파일 생성
# ===================================================================
print("\n" + "=" * 60)
print("[6/6] 제출 파일 생성")
print("=" * 60)

out_dir = os.path.join(try_dir, 'submissions_rev12')
os.makedirs(out_dir, exist_ok=True)
file_count = 0

def save(name, pred_log, desc=""):
    global file_count
    preds = np.expm1(pred_log)
    preds = np.maximum(preds, 0)
    sub = pd.DataFrame({'Id': test_ids, 'SalePrice': preds})
    path = os.path.join(out_dir, f'{name}.csv')
    sub.to_csv(path, index=False)
    # 루트에도 복사
    shutil.copy2(path, os.path.join(try_dir, f'{name}.csv'))
    file_count += 1
    print(f"  {file_count:2d}. {name}.csv | {desc} | 평균: ${preds.mean():,.0f}")

ws, wx, wl = best_w

# OOF 기반
save("sub_blend_optimal_oof", best_w[0]*stacked_test + best_w[1]*xgb_test + best_w[2]*lgb_test,
     f"최적({ws}/{wx}/{wl}) OOF")
save("sub_serigne_oof", 0.70*stacked_test + 0.15*xgb_test + 0.15*lgb_test,
     "Serigne 70/15/15 OOF")
save("sub_stacked_oof", stacked_test, "Stacked OOF")

# Full 기반
save("sub_blend_optimal_full", ws*stacked_full + wx*xgb_full + wl*lgb_full,
     f"최적({ws}/{wx}/{wl}) Full")
save("sub_serigne_full", 0.70*stacked_full + 0.15*xgb_full + 0.15*lgb_full,
     "Serigne 70/15/15 Full")
save("sub_stacked_full", stacked_full, "Stacked Full")

# OOF+Full 평균
save("sub_oof_full_avg",
     0.5*(ws*stacked_test + wx*xgb_test + wl*lgb_test) + 0.5*(ws*stacked_full + wx*xgb_full + wl*lgb_full),
     "OOF+Full 평균")
save("sub_serigne_avg",
     0.5*(0.70*stacked_test + 0.15*xgb_test + 0.15*lgb_test) + 0.5*(0.70*stacked_full + 0.15*xgb_full + 0.15*lgb_full),
     "Serigne OOF+Full 평균")

# 비율 변형
for w1, w2, w3 in [(0.60,0.20,0.20), (0.50,0.25,0.25), (0.80,0.10,0.10)]:
    save(f"sub_blend_{int(w1*100)}_{int(w2*100)}_{int(w3*100)}",
         w1*stacked_full + w2*xgb_full + w3*lgb_full,
         f"Stack{int(w1*100)}/XGB{int(w2*100)}/LGB{int(w3*100)}")

# Boost only
save("sub_boost_50_50", 0.5*xgb_full + 0.5*lgb_full, "XGB+LGB 50/50")
save("sub_xgboost", xgb_full, "XGBoost only")
save("sub_lightgbm", lgb_full, "LightGBM only")

# ===================================================================
# 최종 요약
# ===================================================================
total_time = time.time() - T_GLOBAL
print("\n" + "=" * 60)
print("최종 요약")
print("=" * 60)
print(f"  총 소요 시간: {total_time/60:.1f}분")
print(f"  총 제출 파일: {file_count}개")
print(f"  저장 위치: {out_dir}/")
print(f"\n  OOF RMSE 순위:")
all_scores = {**oof_scores, 'Stacked': stacked_rmse, f'Blend({ws}/{wx}/{wl})': best_rmse, 'Serigne(70/15/15)': serigne_rmse}
for name, score in sorted(all_scores.items(), key=lambda x: x[1]):
    print(f"    {name}: {score:.5f}")
print(f"\n  제출 우선순위:")
print(f"    1. sub_blend_optimal_oof")
print(f"    2. sub_serigne_oof")
print(f"    3. sub_oof_full_avg")
print(f"    4. sub_blend_optimal_full")
