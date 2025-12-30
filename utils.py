import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import shapiro, jarque_bera
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# ================== 1. 数据清洗模块 ==================
def data_cleaning(df):
    """数据清洗：处理缺失值、异常值、数据类型转换"""
    df = df.copy()
    
    # 自动检测和重命名列
    df = auto_rename_columns(df)
    
    # 处理日期列
    df['F10'] = pd.to_datetime(df['F10'], errors='coerce')
    df = df.dropna(subset=['F10'])  # 删除日期解析失败的行
    
    # 处理布尔型列（F7, F8）
    for col in ['F7', 'F8']:
        if col in df.columns:
            # 支持多种布尔型格式
            df[col] = df[col].astype(str).str.upper()
            df[col] = df[col].map({
                'Y': 1, 'YES': 1, 'TRUE': 1, '1': 1, 1: 1,
                'N': 0, 'NO': 0, 'FALSE': 0, '0': 0, 0: 0
            }).fillna(0)
    
    # 处理数值型列
    numeric_cols = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F9']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 删除没有目标变量的行
    df = df.dropna(subset=['T1'])
    
    # 确保销量为正数
    df['T1'] = pd.to_numeric(df['T1'], errors='coerce')
    df = df[df['T1'] > 0]  # 删除非正数销量
    
    # 按日期排序
    df = df.sort_values('F10').reset_index(drop=True)
    return df

def auto_rename_columns(df):
    """自动检测和重命名列以匹配项目要求"""
    column_mapping = {}
    
    # 检测销量列 - 优先匹配更具体的名称
    sales_candidates = [
        ('T1', 1), ('销量', 2), ('销售量', 3), ('sales', 4), 
        ('amount', 5), ('value', 6), ('quantity', 7)
    ]
    for candidate, priority in sales_candidates:
        for col in df.columns:
            if str(col).lower() == candidate.lower():
                column_mapping[col] = 'T1'
                break
        if 'T1' in column_mapping.values():
            break
    
    # 检测日期列 - 优先匹配更具体的名称
    date_candidates = [
        ('F10', 1), ('日期', 2), ('date', 3), ('时间', 4), 
        ('time', 5), ('timestamp', 6)
    ]
    for candidate, priority in date_candidates:
        for col in df.columns:
            if str(col).lower() == candidate.lower() and col not in column_mapping:
                column_mapping[col] = 'F10'
                break
        if 'F10' in column_mapping.values():
            break
    
    # 检测其他特征列
    feature_candidates = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']
    existing_features = [col for col in df.columns if col not in column_mapping.keys()]
    
    for i, col in enumerate(existing_features):
        if i < len(feature_candidates) and col not in ['T1', 'F10']:
            column_mapping[col] = feature_candidates[i]
    
    # 应用重命名
    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"列重命名映射: {column_mapping}")
    
    return df

# ================== 2. 特征工程模块 ==================
def feature_engineering(df):
    df = df.copy()
    df['year'] = df['F10'].dt.year
    df['month'] = df['F10'].dt.month
    df['day'] = df['F10'].dt.day
    df['dayofweek'] = df['F10'].dt.dayofweek
    df['quarter'] = df['F10'].dt.quarter
    df['dayofyear'] = df['F10'].dt.dayofyear
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df['F10'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['F10'].dt.is_month_end.astype(int)
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'T1_lag_{lag}'] = df['T1'].shift(lag)
    for window in [7, 14, 30]:
        df[f'T1_rolling_mean_{window}'] = df['T1'].shift(1).rolling(window=window).mean()
        df[f'T1_rolling_std_{window}'] = df['T1'].shift(1).rolling(window=window).std()
        df[f'T1_rolling_max_{window}'] = df['T1'].shift(1).rolling(window=window).max()
        df[f'T1_rolling_min_{window}'] = df['T1'].shift(1).rolling(window=window).min()
    df['T1_diff_1'] = df['T1'].diff(1)
    df['T1_diff_7'] = df['T1'].diff(7)
    df = df.dropna().reset_index(drop=True)
    return df

# ================== 3. 模型训练模块 ==================
def train_model(X_train, y_train, X_val, y_val):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'max_depth': 6,
        'min_child_samples': 20
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    return model

# ================== 4. 模型评估模块 ==================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'predictions': y_pred
    }

# ================== 5. 预测未来7天 ==================
def predict_next_7_days(model, df, feature_cols, days=7):
    predictions = []
    df_pred = df.copy()
    current_date = df_pred['F10'].max()
    for day in range(1, days + 1):
        next_date = current_date + pd.Timedelta(days=day)
        new_row = df_pred.iloc[-1:].copy()
        new_row['F10'] = next_date
        new_row['year'] = next_date.year
        new_row['month'] = next_date.month
        new_row['day'] = next_date.day
        new_row['dayofweek'] = next_date.dayofweek
        new_row['quarter'] = next_date.quarter
        new_row['dayofyear'] = next_date.dayofyear
        new_row['is_weekend'] = int(next_date.dayofweek >= 5)
        new_row['is_month_start'] = int(next_date.is_month_start)
        new_row['is_month_end'] = int(next_date.is_month_end)
        X_new = new_row[feature_cols]
        pred = model.predict(X_new)[0]
        new_row['T1'] = pred
        df_pred = pd.concat([df_pred, new_row], ignore_index=True)
        df_pred = feature_engineering(df_pred)  # 重算特征
        predictions.append({
            'date': next_date.strftime('%Y-%m-%d'),
            'predicted_sales': round(pred, 2)
        })
    return pd.DataFrame(predictions)

# ================== 6. 残差诊断模块 ==================
def residual_diagnosis(y_true, y_pred):
    residuals = y_true - y_pred
    enterprise_standards = pd.DataFrame({
        '诊断项目': [
            '残差均值接近0',
            '残差方差恒定（无异方差）',
            '残差服从正态分布',
            '残差无自相关（白噪声）'
        ],
        '企标要求': [
            '均值绝对值 < 0.1 * RMSE',
            '残差 vs 预测值散点图无明显模式',
            'Shapiro-Wilk 和 Jarque-Bera 检验 p > 0.05',
            'Ljung-Box 检验 p > 0.05（滞后1-20）'
        ]
    })
    mean_res = np.mean(residuals)
    rmse = np.sqrt(np.mean(residuals**2))
    shapiro_stat, shapiro_p = shapiro(residuals)
    jb_stat, jb_p = jarque_bera(residuals)
    lb_test = sm.stats.acorr_ljungbox(residuals, lags=[20], return_df=True)
    lb_pvalue = lb_test['lb_pvalue'].iloc[-1]
    checks = []
    inconsistencies = []
    if abs(mean_res) < 0.1 * rmse:
        checks.append("是")
        inconsistencies.append("")
    else:
        checks.append("否")
        inconsistencies.append(f"残差均值 {mean_res:.4f}，超过 RMSE 的10%")
    checks.append("是（需目视确认）")
    inconsistencies.append("建议绘制残差 vs 预测值散点图检查异方差")
    if shapiro_p > 0.05 and jb_p > 0.05:
        checks.append("是")
        inconsistencies.append("")
    else:
        checks.append("否")
        inconsistencies.append(f"Shapiro-Wilk p={shapiro_p:.4f}, Jarque-Bera p={jb_p:.4f}")
    if lb_pvalue > 0.05:
        checks.append("是")
        inconsistencies.append("")
    else:
        checks.append("否")
        inconsistencies.append(f"Ljung-Box p={lb_pvalue:.4f}（存在自相关）")
    enterprise_standards['是否与RMS一致'] = checks
    enterprise_standards['不一致的内容'] = inconsistencies
    return enterprise_standards

# ================== 7. 可视化函数 ==================
def plot_sales_trend(df, future_predictions):
    fig = px.line(df, x='F10', y='T1', title='历史销量趋势')
    future_dates = pd.to_datetime(future_predictions['date'])
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions['predicted_sales'], mode='lines+markers', name='未来预测', line=dict(color='green')))
    return fig

def plot_feature_importance(model, feature_cols):
    importance = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importance()}).sort_values('Importance', ascending=False).head(10)
    fig = px.bar(importance, x='Importance', y='Feature', orientation='h', title='Top 10 特征重要性')
    return fig

def plot_residual_distribution(y_true, y_pred):
    residuals = y_true - y_pred
    fig = px.histogram(residuals, nbins=30, title='残差分布图')
    return fig

# ================== 主函数 ==================
def run_prediction(df, days=7):
    df_clean = data_cleaning(df)
    df_features = feature_engineering(df_clean)
    split_date = df_features['F10'].max() - pd.Timedelta(days=30)
    train_df = df_features[df_features['F10'] <= split_date]
    test_df = df_features[df_features['F10'] > split_date]
    feature_cols = [col for col in df_features.columns if col not in ['T1', 'F10']]
    X_train = train_df[feature_cols]
    y_train = train_df['T1']
    X_test = test_df[feature_cols]
    y_test = test_df['T1']
    val_size = int(len(X_train) * 0.1)
    X_train_split = X_train.iloc[:-val_size]
    y_train_split = y_train.iloc[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    model = train_model(X_train_split, y_train_split, X_val, y_val)
    results = evaluate_model(model, X_test, y_test)
    future_predictions = predict_next_7_days(model, df_features, feature_cols, days)
    diagnosis = residual_diagnosis(y_test, results['predictions'])
    return model, results, future_predictions, df_features, feature_cols, y_test, diagnosis