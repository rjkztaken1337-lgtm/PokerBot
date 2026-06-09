#!/usr/bin/env python3
"""
retrain_model.py - дообучение модели на основе обратной связи пользователей.
"""

import sqlite3
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval

ORIGINAL_CSV = "poker_hands_fixed.csv"
DB_PATH = "/Users/user/Desktop/PokerBot/feedback.db"
MODEL_OUT = "poker_model.pkl"
ENCODER_OUT = "poker_encoder.pkl"

# Определение позиции -> число (такое же как в боте)
POS_ORDER = {'BTN':0, 'SB':1, 'BB':2, 'CO':3, 'MP3':4, 'MP2':5, 'MP1':6, 'EP':7}

FEATURE_COLS = [
    'hero_position_num',
    'hero_stack_pre_bb',
    'num_opponents_preflop',
    'is_pair',
    'suited',
    'high_card_rank',
    'low_card_rank',
    'gap',
    'hand_group'
]
TARGET_COL = 'hero_action_preflop'

def load_original_data():
    df = pd.read_csv(ORIGINAL_CSV)
    print("Колонки в исходном CSV:", df.columns.tolist())
    
    # Если нет hero_position_num, создадим из hero_position
    if 'hero_position_num' not in df.columns and 'hero_position' in df.columns:
        df['hero_position_num'] = df['hero_position'].map(POS_ORDER).fillna(7)
    
    # Проверим наличие всех признаков
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Отсутствуют колонки: {missing}")
    
    # Фильтруем действия
    df = df[df[TARGET_COL].notna()]
    df[TARGET_COL] = df[TARGET_COL].str.lower().replace({
        'folds': 'fold', 'raises': 'raise', 'calls': 'call', 'checks': 'check'
    })
    df = df[df[TARGET_COL].isin(['fold','call','raise','check'])]
    
    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]
    return X, y

def load_feedback_data():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT features, predicted_action FROM predictions WHERE user_feedback = 'yes'"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if df.empty:
        return None, None
    features_list = []
    for feat_str in df['features']:
        try:
            feat = literal_eval(feat_str)
            features_list.append(feat)
        except:
            continue
    if not features_list:
        return None, None
    X_fb = pd.DataFrame(features_list, columns=FEATURE_COLS)
    y_fb = df['predicted_action'].str.lower().replace({
        'folds': 'fold', 'raises': 'raise', 'calls': 'call', 'checks': 'check'
    })
    return X_fb, y_fb

def main():
    print("1. Загрузка исходных данных...")
    X_orig, y_orig = load_original_data()
    print(f"   Исходных рук: {len(X_orig)}")
    
    print("2. Загрузка данных обратной связи (feedback='yes')...")
    X_fb, y_fb = load_feedback_data()
    if X_fb is not None and len(X_fb) > 0:
        print(f"   Добавлено рук с положительной обратной связью: {len(X_fb)}")
        X_combined = pd.concat([X_orig, X_fb], ignore_index=True)
        y_combined = pd.concat([y_orig, y_fb], ignore_index=True)
    else:
        print("   Нет новых данных. Обучение только на исходных.")
        X_combined = X_orig
        y_combined = y_orig
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y_combined)
    
    print("3. Обучение модели RandomForest...")
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_combined, y_enc)
    
    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_combined)
    acc = accuracy_score(y_enc, y_pred)
    print(f"   Точность на обучающей выборке: {acc:.3f}")
    
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(model, f)
    with open(ENCODER_OUT, 'wb') as f:
        pickle.dump(le, f)
    print(f"4. Модель сохранена в {MODEL_OUT}")
    print(f"   Энкодер сохранён в {ENCODER_OUT}")

if __name__ == "__main__":
    main()