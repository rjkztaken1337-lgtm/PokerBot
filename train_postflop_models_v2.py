#!/usr/bin/env python3
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = [
    'hero_position_num', 'hero_stack_pre_bb', 'preflop_opponents',
    'is_pair', 'suited', 'high_card_rank', 'low_card_rank', 'gap', 'hand_group',
    'opponents'
]

def train_for_street(df, street):
    print(f"\nОбучение для {street.upper()}...")
    df_street = df[df['street'] == street].copy()
    if df_street.empty:
        print(f"Нет данных для {street}")
        return
    # Проверяем, что все необходимые колонки есть
    missing = [c for c in FEATURE_COLS if c not in df_street.columns]
    if missing:
        print(f"Пропущены колонки: {missing}")
        return
    X = df_street[FEATURE_COLS].fillna(0)
    y = df_street['action'].str.lower().replace({
        'folds':'fold', 'calls':'call', 'bets':'bet', 'raises':'raise', 'checks':'check'
    })
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X, y_enc)
    acc = model.score(X, y_enc)
    print(f"Точность на обучении: {acc:.3f}")
    with open(f"{street}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{street}_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print(f"Модель {street} сохранена.")

def main():
    df = pd.read_csv("postflop_dataset.csv")
    print(f"Всего записей: {len(df)}")
    # Посмотрим доступные улицы
    print("Улицы в данных:", df['street'].unique())
    for street in ['flop', 'turn', 'river']:
        train_for_street(df, street)

if __name__ == "__main__":
    main()