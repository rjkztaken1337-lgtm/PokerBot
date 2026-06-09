import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

CSV_PATH = "poker_hands_fixed.csv"
MODEL_OUT = "poker_model.pkl"
ENCODER_OUT = "poker_encoder.pkl"

POS_ORDER = {'BTN':0,'SB':1,'BB':2,'CO':3,'MP3':4,'MP2':5,'MP1':6,'EP':7}
FEATURE_COLS = ['hero_position_num', 'hero_stack_pre_bb', 'num_opponents_preflop',
                'is_pair', 'suited', 'high_card_rank', 'low_card_rank', 'gap', 'hand_group']
TARGET_COL = 'hero_action_preflop'

df = pd.read_csv(CSV_PATH)
if 'hero_position_num' not in df.columns and 'hero_position' in df.columns:
    df['hero_position_num'] = df['hero_position'].map(POS_ORDER).fillna(7)

df = df[df[TARGET_COL].notna()]
df[TARGET_COL] = df[TARGET_COL].str.lower().replace({'folds':'fold','raises':'raise','calls':'call','checks':'check'})
df = df[df[TARGET_COL].isin(['fold','call','raise','check'])]

X = df[FEATURE_COLS].fillna(0)
y = df[TARGET_COL]

le = LabelEncoder()
y_enc = le.fit_transform(y)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X, y_enc)

with open(MODEL_OUT, 'wb') as f:
    pickle.dump(model, f)
with open(ENCODER_OUT, 'wb') as f:
    pickle.dump(le, f)

print("Model retrained successfully inside container")