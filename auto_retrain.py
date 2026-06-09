#!/usr/bin/env python3
import sqlite3
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

# Пути
ORIGINAL_CSV = "poker_hands_fixed.csv"   # исходный датасет
DB_PATH = "feedback.db"
MODEL_OUT = "poker_model.pkl"
ENCODER_OUT = "poker_encoder.pkl"

# Признаки (те же, что используются в боте)
FEATURE_COLS = [
    'hero_position_num', 'hero_stack_pre_bb', 'num_opponents_preflop',
    'is_pair', 'suited', 'high_card_rank', 'low_card_rank', 'gap', 'hand_group'
]
TARGET_COL = 'hero_action_preflop'
POS_ORDER = {'BTN':0,'SB':1,'BB':2,'CO':3,'MP3':4,'MP2':5,'MP1':6,'EP':7}

def load_original_data():
    df = pd.read_csv(ORIGINAL_CSV)
    if 'hero_position_num' not in df.columns and 'hero_position' in df.columns:
        df['hero_position_num'] = df['hero_position'].map(POS_ORDER).fillna(7)
    df = df[df[TARGET_COL].notna()]
    df[TARGET_COL] = df[TARGET_COL].str.lower().replace({
        'folds':'fold','raises':'raise','calls':'call','checks':'check'
    })
    df = df[df[TARGET_COL].isin(['fold','call','raise','check'])]
    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]
    return X, y

def load_feedback_data():
    if not os.path.exists(DB_PATH):
        return None, None
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT features, predicted_action FROM predictions WHERE user_feedback = 'yes'", conn)
    except:
        conn.close()
        return None, None
    conn.close()
    if df.empty:
        return None, None
    features_list = []
    for feat_str in df['features']:
        try:
            feat = eval(feat_str)  # безопасно, т.к. данные свои
            features_list.append(feat)
        except:
            continue
    if not features_list:
        return None, None
    X_fb = pd.DataFrame(features_list, columns=FEATURE_COLS)
    y_fb = df['predicted_action'].str.lower().replace({
        'folds':'fold','raises':'raise','calls':'call','checks':'check'
    })
    return X_fb, y_fb

def main():
    print("Загрузка исходных данных...")
    X_orig, y_orig = load_original_data()
    print(f"Исходных рук: {len(X_orig)}")
    
    print("Загрузка обратной связи (feedback=yes)...")
    X_fb, y_fb = load_feedback_data()
    if X_fb is not None and len(X_fb) > 0:
        print(f"Добавлено рук с положительной обратной связью: {len(X_fb)}")
        X_combined = pd.concat([X_orig, X_fb], ignore_index=True)
        y_combined = pd.concat([y_orig, y_fb], ignore_index=True)
    else:
        print("Нет новых отзывов. Оставляем исходную модель.")
        X_combined = X_orig
        y_combined = y_orig
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y_combined)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_combined, y_enc)
    acc = accuracy_score(y_enc, model.predict(X_combined))
    print(f"Точность на обучающей выборке: {acc:.3f}")
    
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(model, f)
    with open(ENCODER_OUT, 'wb') as f:
        pickle.dump(le, f)
    print("Модель и энкодер сохранены.")

if __name__ == "__main__":
    main()