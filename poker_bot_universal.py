#!/usr/bin/env python3
import os
import re
import pickle
import logging
import sqlite3
import subprocess
import sys
import numpy as np
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes

# ---------- Настройка ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- Токен и админ ----------
TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_TOKEN not set")
ADMIN_ID = int(os.environ.get("ADMIN_ID", 0))

# ---------- Глобальные модели ----------
model = None
encoder = None
flop_model = None
flop_encoder = None
turn_model = None
turn_encoder = None
river_model = None
river_encoder = None

def load_model_files():
    global model, encoder
    try:
        with open("poker_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("poker_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        logger.info("Префлоп-модель загружена")
        return True
    except Exception as e:
        logger.error(f"Ошибка загрузки префлоп-модели: {e}")
        model = None
        encoder = None
        return False

def load_postflop_models():
    global flop_model, flop_encoder, turn_model, turn_encoder, river_model, river_encoder
    try:
        if os.path.exists("flop_model.pkl"):
            with open("flop_model.pkl", "rb") as f:
                flop_model = pickle.load(f)
            with open("flop_encoder.pkl", "rb") as f:
                flop_encoder = pickle.load(f)
            logger.info("Флоп-модель загружена")
        if os.path.exists("turn_model.pkl"):
            with open("turn_model.pkl", "rb") as f:
                turn_model = pickle.load(f)
            with open("turn_encoder.pkl", "rb") as f:
                turn_encoder = pickle.load(f)
            logger.info("Тёрн-модель загружена")
        if os.path.exists("river_model.pkl"):
            with open("river_model.pkl", "rb") as f:
                river_model = pickle.load(f)
            with open("river_encoder.pkl", "rb") as f:
                river_encoder = pickle.load(f)
            logger.info("Ривер-модель загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки постфлоп-моделей: {e}")

load_model_files()
load_postflop_models()

# ---------- База данных ----------
DB_PATH = "feedback.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            hand_text TEXT,
            features TEXT,
            predicted_action TEXT,
            user_feedback TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(user_id, hand_text, features, predicted_action, feedback=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (user_id, hand_text, features, predicted_action, user_feedback, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, hand_text[:1000], str(features), predicted_action, feedback, datetime.now().isoformat()))
    conn.commit()
    pred_id = c.lastrowid
    conn.close()
    return pred_id

def update_feedback(pred_id, feedback):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE predictions SET user_feedback = ? WHERE id = ?', (feedback, pred_id))
    conn.commit()
    conn.close()

init_db()

# ---------- Константы ----------
RANK_ORDER = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
RANK_TO_CHAR = {2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'T',11:'J',12:'Q',13:'K',14:'A'}
POS_ORDER = {'BTN':0,'SB':1,'BB':2,'CO':3,'MP3':4,'MP2':5,'MP1':6,'EP':7}
rank_to_letter = {v:k for k,v in RANK_ORDER.items()}

def determine_position(hero_seat, button_seat, total_seats=9):
    diff = (hero_seat - button_seat) % total_seats
    if diff == 0: return "BTN"
    if diff == 1: return "SB"
    if diff == 2: return "BB"
    if diff == total_seats - 1: return "CO"
    if diff == total_seats - 2: return "MP3"
    if diff == total_seats - 3: return "MP2"
    if diff == total_seats - 4: return "MP1"
    return "EP"

def hand_features(cards_str):
    if not cards_str or len(cards_str.strip().split()) != 2:
        return {'is_pair':0,'suited':0,'high_card_rank':0,'low_card_rank':0,'gap':-1,'hand_group':0}
    c1, c2 = cards_str.strip().split()
    r1 = RANK_ORDER.get(c1[0], 0)
    r2 = RANK_ORDER.get(c2[0], 0)
    suited = 1 if (c1[1] == c2[1]) else 0
    high = max(r1, r2)
    low = min(r1, r2)
    is_pair = int(r1 == r2)
    gap = (high - low) - 1 if not is_pair else -1
    if is_pair:
        group = 10 if high >= 10 else (8 if high >= 8 else 6)
    else:
        if suited and high >= 12 and gap <= 2:
            group = 9
        elif suited and high >= 10:
            group = 7
        elif not suited and high >= 12 and gap == 0:
            group = 6
        else:
            group = 3
    return {'is_pair':is_pair, 'suited':suited, 'high_card_rank':high, 'low_card_rank':low,
            'gap':gap, 'hand_group':group}

# ---------- Парсер с точным расчётом банка ----------
def parse_hand_advanced(content):
    result = {
        'hero_seat': None,
        'hero_position': None,
        'hero_hole_cards': None,
        'big_blind': 1.0,
        'hero_stack_pre_bb': 0.0,
        'preflop_action': None,
        'preflop_opponents': 0,
        'flop_action': None,
        'flop_opponents': 0,
        'flop_cbet': 0,
        'flop_cards': None,
        'flop_pot': 0.0,
        'flop_bet': 0.0,
        'turn_action': None,
        'turn_opponents': 0,
        'turn_cards': None,
        'turn_pot': 0.0,
        'turn_bet': 0.0,
        'river_action': None,
        'river_opponents': 0,
        'river_cards': None,
        'river_pot': 0.0,
        'river_bet': 0.0,
        'board_cards': None,
        'showdown_hero_hand': None,
        'showdown_hero_rank': None,
        'showdown_villain_hand': None,
        'showdown_villain_rank': None,
        'hero_won': None,
        'hero_win_amount': 0.0,
    }
    # Базовая информация
    blinds = re.search(r'\$([\d\.]+)/\$([\d\.]+)', content)
    if blinds:
        result['big_blind'] = float(blinds.group(2))
    hero_seat_match = re.search(r'Seat\s+(\d+):\s*Hero', content, re.IGNORECASE)
    if not hero_seat_match:
        return None
    hero_seat = int(hero_seat_match.group(1))
    result['hero_seat'] = hero_seat
    button_match = re.search(r'Seat\s+#?(\d+)\s+is the button', content, re.IGNORECASE)
    if button_match:
        button_seat = int(button_match.group(1))
        result['hero_position'] = determine_position(hero_seat, button_seat, 9)
    cards_match = re.search(r'Hero\s+\[([2-9TJQKA][cdhs] [2-9TJQKA][cdhs])\]', content)
    if not cards_match:
        cards_match = re.search(r'Dealt to Hero\s+\[([2-9TJQKA][cdhs] [2-9TJQKA][cdhs])\]', content)
    if cards_match:
        result['hero_hole_cards'] = cards_match.group(1)
    else:
        return None
    stack_match = re.search(r'Seat\s+%d:\s+Hero\s+\(\$([\d\.]+)' % hero_seat, content)
    if stack_match and result['big_blind'] > 0:
        result['hero_stack_pre_bb'] = float(stack_match.group(1)) / result['big_blind']

    # Карты стола
    flop_match = re.search(r'FLOP.*?\[([^]]+)\]', content, re.IGNORECASE)
    if flop_match:
        result['flop_cards'] = flop_match.group(1)
    turn_match = re.search(r'TURN.*?\[([^]]+)\](?:.*?\[([^]]+)\])?', content, re.IGNORECASE)
    if turn_match:
        if turn_match.group(2):
            result['turn_cards'] = turn_match.group(2)
        else:
            result['turn_cards'] = turn_match.group(1)
    river_match = re.search(r'RIVER.*?\[([^]]+)\](?:.*?\[([^]]+)\])?', content, re.IGNORECASE)
    if river_match:
        if river_match.group(2):
            result['river_cards'] = river_match.group(2)
        else:
            result['river_cards'] = river_match.group(1)
    # Все карты стола
    board_cards = []
    if result['flop_cards']:
        board_cards.extend(result['flop_cards'].split())
    if result['turn_cards']:
        board_cards.append(result['turn_cards'])
    if result['river_cards']:
        board_cards.append(result['river_cards'])
    result['board_cards'] = " ".join(board_cards) if board_cards else None

    # Функция для извлечения суммы из строки действия (с учётом рейзов как разницы)
    def get_amount(line):
        # posts small blind $0.01
        m = re.search(r'posts (?:small blind|big blind|ante)\s+\$([\d\.]+)', line, re.IGNORECASE)
        if m:
            return float(m.group(1))
        # calls $0.23
        m = re.search(r'calls?\s+\$([\d\.]+)', line, re.IGNORECASE)
        if m:
            return float(m.group(1))
        # bets $0.63
        m = re.search(r'bets\s+\$([\d\.]+)', line, re.IGNORECASE)
        if m:
            return float(m.group(1))
        # raises $0.23 to $0.34
        m = re.search(r'raises\s+\$([\d\.]+)\s+to\s+\$([\d\.]+)', line, re.IGNORECASE)
        if m:
            # Добавляем только разницу (то, что фактически добавилось в банк)
            return float(m.group(2)) - float(m.group(1))
        # raises $0.23 (без "to")
        m = re.search(r'raises\s+\$([\d\.]+)', line, re.IGNORECASE)
        if m:
            # Это может быть рейз без указания предыдущей ставки – добавляем всю сумму
            return float(m.group(1))
        return 0.0

    # Разбиваем текст на блоки по улицам
    pre_block = re.search(r'(?:HOLE CARDS.*?)(?=\*\*\*?\s*FLOP|\Z)', content, re.DOTALL | re.IGNORECASE)
    pre_text = pre_block.group(0) if pre_block else ""
    flop_block = re.search(r'(?:FLOP.*?)(?=\*\*\*?\s*TURN|\Z)', content, re.DOTALL | re.IGNORECASE)
    flop_text = flop_block.group(0) if flop_block else ""
    turn_block = re.search(r'(?:TURN.*?)(?=\*\*\*?\s*RIVER|\Z)', content, re.DOTALL | re.IGNORECASE)
    turn_text = turn_block.group(0) if turn_block else ""
    river_block = re.search(r'(?:RIVER.*?)(?=\*\*\*?\s*SHOW DOWN|\Z)', content, re.DOTALL | re.IGNORECASE)
    river_text = river_block.group(0) if river_block else ""

    # --- Префлоп ---
    pot = 0.0
    for line in pre_text.split('\n'):
        pot += get_amount(line)
    pre_action_match = re.search(r'Hero:\s*(folds|checks|calls|bets|raises)', pre_text, re.IGNORECASE)
    if pre_action_match:
        result['preflop_action'] = pre_action_match.group(1).lower()
    pre_players = set(re.findall(r'([A-Za-z0-9_]+):\s*(?:folds|raises|calls|bets|checks)', pre_text))
    pre_players.discard('Hero')
    result['preflop_opponents'] = len(pre_players)

    # --- Флоп ---
    for line in flop_text.split('\n'):
        pot += get_amount(line)
    result['flop_pot'] = pot
    flop_action_match = re.search(r'Hero:\s*(folds|checks|calls|bets|raises)', flop_text, re.IGNORECASE)
    if flop_action_match:
        result['flop_action'] = flop_action_match.group(1).lower()
    flop_players = set(re.findall(r'([A-Za-z0-9_]+):\s*(?:folds|raises|calls|bets|checks)', flop_text))
    flop_players.discard('Hero')
    result['flop_opponents'] = len(flop_players)
    flop_bet_match = re.search(r'Hero:\s+bets\s+\$([\d\.]+)', flop_text, re.IGNORECASE)
    if flop_bet_match:
        result['flop_bet'] = float(flop_bet_match.group(1))
    if result['flop_action'] in ['bets', 'raises']:
        result['flop_cbet'] = 1

    # --- Тёрн ---
    for line in turn_text.split('\n'):
        pot += get_amount(line)
    result['turn_pot'] = pot
    turn_action_match = re.search(r'Hero:\s*(folds|checks|calls|bets|raises)', turn_text, re.IGNORECASE)
    if turn_action_match:
        result['turn_action'] = turn_action_match.group(1).lower()
    turn_players = set(re.findall(r'([A-Za-z0-9_]+):\s*(?:folds|raises|calls|bets|checks)', turn_text))
    turn_players.discard('Hero')
    result['turn_opponents'] = len(turn_players)
    turn_bet_match = re.search(r'Hero:\s+bets\s+\$([\d\.]+)', turn_text, re.IGNORECASE)
    if turn_bet_match:
        result['turn_bet'] = float(turn_bet_match.group(1))

    # --- Ривер ---
    for line in river_text.split('\n'):
        pot += get_amount(line)
    result['river_pot'] = pot
    river_action_match = re.search(r'Hero:\s*(folds|checks|calls|bets|raises)', river_text, re.IGNORECASE)
    if river_action_match:
        result['river_action'] = river_action_match.group(1).lower()
    river_players = set(re.findall(r'([A-Za-z0-9_]+):\s*(?:folds|raises|calls|bets|checks)', river_text))
    river_players.discard('Hero')
    result['river_opponents'] = len(river_players)
    river_bet_match = re.search(r'Hero:\s+bets\s+\$([\d\.]+)', river_text, re.IGNORECASE)
    if river_bet_match:
        result['river_bet'] = float(river_bet_match.group(1))

    # ---------- Шоудаун ----------
    # Hero
    hero_show = re.search(r'Hero[:\s]+shows?ed?\s+\[([^]]+)\]\s*\(([^)]+)\)', content, re.IGNORECASE)
    if not hero_show:
        hero_show = re.search(r'Hero[:\s]+shows?ed?\s+\[([^]]+)\][:\s]+and (?:won|lost) with\s+(.+)', content, re.IGNORECASE)
    if hero_show:
        result['showdown_hero_hand'] = hero_show.group(1).strip()
        result['showdown_hero_rank'] = hero_show.group(2).strip()
    # Оппонент
    villain_show = re.search(r'Seat\s+\d+:\s+[A-Za-z0-9_]+\s+shows?ed?\s+\[([^]]+)\]\s+and won [^$]*\$(?:[\d\.]+).*?with\s+(.+)', content, re.IGNORECASE)
    if not villain_show:
        villain_show = re.search(r'Seat\s+\d+:\s+[A-Za-z0-9_]+\s+shows?ed?\s+\[([^]]+)\]\s*\(([^)]+)\)', content, re.IGNORECASE)
    if villain_show and 'Hero' not in villain_show.group(0):
        result['showdown_villain_hand'] = villain_show.group(1).strip()
        result['showdown_villain_rank'] = villain_show.group(2).strip()
    
    # Выигрыш Hero
    hero_won_match = re.search(r'Hero\s+collected\s+\$([\d\.]+)', content, re.IGNORECASE)
    if not hero_won_match:
        hero_won_match = re.search(r'Seat\s+%d:\s+Hero\s+.*?\s+won\s+\$([\d\.]+)' % hero_seat, content, re.IGNORECASE)
    if hero_won_match:
        result['hero_won'] = True
        result['hero_win_amount'] = float(hero_won_match.group(1))
    else:
        if re.search(r'Hero\s+lost', content, re.IGNORECASE) or re.search(r'Seat\s+%d:\s+Hero\s+.*?\s+lost' % hero_seat, content, re.IGNORECASE):
            result['hero_won'] = False
        else:
            result['hero_won'] = None
    return result

def calculate_draws_and_odds(hero_cards, board_cards, street, pot_size=0, bet=0):
    if not hero_cards or not board_cards:
        return ""
    all_cards = hero_cards.split() + board_cards.split()
    hero_suits = [c[1] for c in hero_cards.split()]
    draws = []
    total_outs = 0
    if hero_suits[0] == hero_suits[1]:
        suit = hero_suits[0]
        suited_on_board = sum(1 for c in board_cards.split() if c[1] == suit)
        total_suited = suited_on_board + 2
        if total_suited == 4:
            outs = 9
            draws.append(f"💧 Флеш-дро ({outs} аутов)")
            total_outs += outs
        elif total_suited == 3:
            draws.append("💧 Бэкдор-флеш (нужны две карты)")
    all_ranks = [RANK_ORDER.get(c[0], 0) for c in all_cards]
    sorted_ranks = sorted(set(all_ranks))
    for i in range(len(sorted_ranks) - 3):
        if sorted_ranks[i+3] - sorted_ranks[i] <= 4:
            draws.append("📏 Стрит-дро (8 аутов)")
            total_outs += 8
            break
    hero_ranks = [RANK_ORDER.get(c[0], 0) for c in hero_cards.split()]
    board_ranks = [RANK_ORDER.get(c[0], 0) for c in board_cards.split()]
    has_pair = any(r in board_ranks for r in hero_ranks) or (hero_ranks[0] == hero_ranks[1])
    if has_pair:
        draws.append("🎯 Есть пара на доске – можно улучшить до трипса (2 аута)")
        total_outs += 2
    equity = 0
    if total_outs > 0:
        if street == 'flop':
            equity = min(total_outs * 4, 100)
        elif street == 'turn':
            equity = min(total_outs * 2, 100)
        else:
            equity = 0
        draws.append(f"📈 Эквити ≈ {equity}% (примерно)")
    else:
        draws.append("🔹 Нет сильных дро, рука слабая")
    odds_comment = ""
    if pot_size > 0 and bet > 0:
        pot_odds = bet / (pot_size + bet) * 100
        odds_comment = f"\n💰 *Шансы банка:* {pot_odds:.1f}% (ставка ${bet:.2f} к банку ${pot_size:.2f})"
        if total_outs > 0 and equity > pot_odds:
            odds_comment += " → колл выгоден (эквити выше шансов банка)"
        elif total_outs > 0:
            odds_comment += " → колл невыгоден"
    return "\n".join(draws) + odds_comment

def analyze_board_texture(board_cards):
    if not board_cards:
        return ""
    cards = board_cards.split()
    suits = [c[1] for c in cards]
    ranks = [c[0] for c in cards]
    rank_values = [RANK_ORDER.get(r, 0) for r in ranks]
    is_monotone = len(set(suits)) == 1
    is_paired = len(set(ranks)) < len(ranks)
    suit_counts = {s: suits.count(s) for s in set(suits)}
    flush_draw = any(count >= 2 for count in suit_counts.values())
    sorted_ranks = sorted(rank_values)
    straight_possible = False
    if not is_paired and max(sorted_ranks) - min(sorted_ranks) <= 4:
        straight_possible = True
    if is_monotone:
        texture = "💧 Монотонная доска – высокий риск флеша."
    elif is_paired:
        texture = "🪵 Парная доска – опасность фулл-хауса или трипса."
    elif straight_possible:
        texture = "📏 Координированная доска (возможен стрит)."
    elif flush_draw:
        texture = "💧 Есть флеш-дро."
    else:
        texture = "🍂 Сухая доска."
    if flush_draw and not is_monotone:
        texture += " Будьте осторожны: возможен флеш на тёрне/ривере."
    if straight_possible and not is_paired:
        texture += " Много стрит-дро."
    return texture

def get_postflop_prediction(parsed, street):
    if street == 'flop' and flop_model is None:
        return None, None, None
    if street == 'turn' and turn_model is None:
        return None, None, None
    if street == 'river' and river_model is None:
        return None, None, None

    card_feats = hand_features(parsed['hero_hole_cards'])
    pos_num = POS_ORDER.get(parsed['hero_position'], 7)
    stack_bb = parsed['hero_stack_pre_bb']
    preflop_opp = parsed.get('preflop_opponents', 0)
    opponents = parsed.get(f'{street}_opponents', 0)
    features = {
        'hero_position_num': pos_num,
        'hero_stack_pre_bb': stack_bb,
        'preflop_opponents': preflop_opp,
        'is_pair': card_feats['is_pair'],
        'suited': card_feats['suited'],
        'high_card_rank': card_feats['high_card_rank'],
        'low_card_rank': card_feats['low_card_rank'],
        'gap': card_feats['gap'],
        'hand_group': card_feats['hand_group'],
        'opponents': opponents,
    }
    if street == 'turn':
        prev_action = parsed.get('flop_action', 'none')
        features['flop_action'] = prev_action
    elif street == 'river':
        prev_action = parsed.get('turn_action', 'none')
        features['turn_action'] = prev_action

    import pandas as pd
    df = pd.DataFrame([features])
    if street == 'turn':
        df = pd.get_dummies(df, columns=['flop_action'], prefix='flop')
    elif street == 'river':
        df = pd.get_dummies(df, columns=['turn_action'], prefix='turn')
    if street == 'flop':
        model = flop_model
        encoder = flop_encoder
        expected_cols = flop_model.feature_names_in_
    elif street == 'turn':
        model = turn_model
        encoder = turn_encoder
        expected_cols = turn_model.feature_names_in_
    else:
        model = river_model
        encoder = river_encoder
        expected_cols = river_model.feature_names_in_
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[expected_cols].fillna(0).values
    pred_enc = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    confidence = np.max(probs)
    action = encoder.inverse_transform([pred_enc])[0]
    return action, confidence, probs

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🃏 Отправьте мне текстовый файл (.txt) с раздачей или просто вставьте текст.\n"
        "Поддерживаются форматы: PokerStars, GG Poker, PartyPoker.\n\n"
        "📌 Команды:\n"
        "/stats — как пользоваться\n"
        "/about — о боте\n"
        "/profile — ваш профиль\n"
        "/analysis — детальный разбор последней руки\n"
        "/explain — объяснение последнего предсказания\n"
        "/flop — предсказание на флопе\n"
        "/turn — предсказание на тёрне\n"
        "/river — предсказание на ривере\n"
        "/feedback — отзыв\n"
        "/terms — условия\n"
        "/reload — перезагрузить модель (админ)\n"
        "/retrain_now — переобучить модель (админ)"
    )

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🎯 *Как пользоваться ботом*\n\n"
        "1️⃣ Скопируйте историю раздачи из покер-рума (PokerStars, GG Poker, PartyPoker).\n"
        "2️⃣ Отправьте её боту как текстовый файл или просто вставьте в чат.\n"
        "3️⃣ Бот найдёт вашу позицию, карты, стек и количество оппонентов.\n"
        "4️⃣ Получите предсказание: 🤚 Fold, 📞 Call, 📈 Raise или ✅ Check.\n"
        "5️⃣ Оцените точность кнопками ✅/❌ – это поможет улучшить модель.\n\n"
        "✨ *Дополнительно:*\n"
        "/analysis – подробный разбор последней руки (включая анализ дро и шансов банка)\n"
        "/explain – понятное объяснение, почему бот дал такой совет\n"
        "/flop, /turn, /river – предсказания на соответствующих улицах (если есть данные)"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 *Poker Oracle Bot v3.4* (оптимизирован расчёт банка, улучшен итог раздачи)\n\n"
        "🧠 *Модели:* Random Forest для префлопа, флопа, тёрна, ривера.\n"
        "📊 *Признаки:* позиция, сила руки, стек, оппоненты, предыдущие действия.\n"
        "♠️ *Поддерживаемые румы:* PokerStars, GG Poker, PartyPoker.\n"
        "📈 *Функции:* предсказание действий, сбор обратной связи, переобучение, объяснение решений, оценка действий пользователя.\n"
        "🃏 *Новое:* отображение всех пяти карт стола, результат внизу итога.\n\n"
        "© 2026 | Сделано с любовью к покеру и AI"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    predictions = context.user_data.get('predictions', 0)
    last = context.user_data.get('last_hand_info')
    last_action = last['action'].upper() if last else 'нет'
    text = f"🆔 *Ваш профиль*\n• Имя: {user.first_name}\n• ID: {user.id}\n\n📊 *Статистика*\n• Предсказаний: {predictions}\n• Последнее действие: {last_action}"
    await update.message.reply_text(text, parse_mode='Markdown')

async def analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    last = context.user_data.get('last_hand_info')
    parsed = context.user_data.get('last_parsed_hand')
    if not last or not parsed:
        await update.message.reply_text("❌ Нет данных о последней руке. Сначала отправьте раздачу.")
        return

    text = (
        f"🔍 *Детальный разбор последней руки*\n\n"
        f"🃏 *Карты:* {last['cards']}\n"
        f"📌 *Позиция:* {last['position']}\n"
        f"💰 *Стек в BB:* {last['stack_bb']:.1f}\n"
        f"👥 *Оппонентов на префлопе:* {last['opponents']}\n\n"
        f"📊 *Признаки:*\n"
        f"• Пара: {'да' if last['is_pair'] else 'нет'}\n"
        f"• Одномастные: {'да' if last['suited'] else 'нет'}\n"
        f"• Старшая карта: {last['high_card']}\n"
        f"• Младшая карта: {last['low_card']}\n"
        f"• Разрыв (gap): {last['gap']}\n"
        f"• Группа руки: {last['hand_group']}\n\n"
        f"🤖 *Предсказание на префлопе:* **{last['action'].upper()}** (уверенность {last['confidence']:.1%})\n"
    )
    user_preflop = parsed.get('preflop_action')
    if user_preflop:
        if user_preflop == last['action']:
            text += f"\n✅ *Префлоп:* Вы {user_preflop.upper()}. Действие совпадает с рекомендацией модели.\n"
        else:
            text += f"\n⚠️ *Префлоп:* Вы {user_preflop.upper()}, но модель рекомендовала {last['action'].upper()} (уверенность {last['confidence']:.1%}).\n"

    if last.get('flop_cards'):
        texture = analyze_board_texture(last['flop_cards'])
        text += f"\n♣️ *Флоп:* {last['flop_cards']}\n   *Текстура:* {texture}"
        pot = parsed.get('flop_pot', 0)
        bet = parsed.get('flop_bet', 0)
        draws_analysis = calculate_draws_and_odds(last['cards'], last['flop_cards'], 'flop', pot, bet)
        if draws_analysis:
            text += f"\n   *Анализ:* {draws_analysis}"
        if last.get('flop_action'):
            text += f"\n   *Ваше действие:* {last['flop_action'].upper()}, оппонентов — {last['flop_opponents']}"
            if last.get('flop_cbet'):
                text += " (конт-бет ✅)"
            flop_pred, flop_conf, _ = get_postflop_prediction(parsed, 'flop')
            if flop_pred:
                if flop_pred == last['flop_action']:
                    text += f"\n   ✅ *Оценка:* Ваше действие совпадает с рекомендацией модели ({flop_pred.upper()}, уверенность {flop_conf:.1%})."
                else:
                    text += f"\n   ⚠️ *Оценка:* Модель рекомендовала {flop_pred.upper()} (уверенность {flop_conf:.1%}), а вы сделали {last['flop_action'].upper()}."
    if last.get('turn_cards'):
        texture = analyze_board_texture(last['turn_cards'])
        text += f"\n♦️ *Тёрн:* {last['turn_cards']}\n   *Текстура:* {texture}"
        pot = parsed.get('turn_pot', 0)
        bet = parsed.get('turn_bet', 0)
        draws_analysis = calculate_draws_and_odds(last['cards'], last['turn_cards'], 'turn', pot, bet)
        if draws_analysis:
            text += f"\n   *Анализ:* {draws_analysis}"
        if last.get('turn_action'):
            text += f"\n   *Ваше действие:* {last['turn_action'].upper()}, оппонентов — {last['turn_opponents']}"
            turn_pred, turn_conf, _ = get_postflop_prediction(parsed, 'turn')
            if turn_pred:
                if turn_pred == last['turn_action']:
                    text += f"\n   ✅ *Оценка:* Ваше действие совпадает с рекомендацией модели ({turn_pred.upper()}, уверенность {turn_conf:.1%})."
                else:
                    text += f"\n   ⚠️ *Оценка:* Модель рекомендовала {turn_pred.upper()} (уверенность {turn_conf:.1%}), а вы сделали {last['turn_action'].upper()}."
    if last.get('river_cards'):
        texture = analyze_board_texture(last['river_cards'])
        text += f"\n♥️ *Ривер:* {last['river_cards']}\n   *Текстура:* {texture}"
        pot = parsed.get('river_pot', 0)
        bet = parsed.get('river_bet', 0)
        draws_analysis = calculate_draws_and_odds(last['cards'], last['river_cards'], 'river', pot, bet)
        if draws_analysis:
            text += f"\n   *Анализ:* {draws_analysis}"
        if last.get('river_action'):
            text += f"\n   *Ваше действие:* {last['river_action'].upper()}, оппонентов — {last['river_opponents']}"
            river_pred, river_conf, _ = get_postflop_prediction(parsed, 'river')
            if river_pred:
                if river_pred == last['river_action']:
                    text += f"\n   ✅ *Оценка:* Ваше действие совпадает с рекомендацией модели ({river_pred.upper()}, уверенность {river_conf:.1%})."
                else:
                    text += f"\n   ⚠️ *Оценка:* Модель рекомендовала {river_pred.upper()} (уверенность {river_conf:.1%}), а вы сделали {last['river_action'].upper()}."

    text += "\n🏆 *ИТОГ РАЗДАЧИ*\n"
    if parsed.get('board_cards'):
        text += f"🃏 *Доска:* {parsed['board_cards']}\n"
    if parsed.get('showdown_hero_rank'):
        text += f"🃏 *Ваша рука:* {parsed['showdown_hero_hand']} – {parsed['showdown_hero_rank']}\n"
    if parsed.get('showdown_villain_rank'):
        text += f"🃏 *Рука оппонента:* {parsed['showdown_villain_hand']} – {parsed['showdown_villain_rank']}\n"
    if parsed.get('hero_won') is True:
        text += f"✅ *Победа Hero!* Выигрыш: ${parsed.get('hero_win_amount', 0):.2f}\n"
    elif parsed.get('hero_won') is False:
        text += "❌ *Поражение Hero*\n"
    else:
        text += "⚠️ *Результат:* Раздача завершена без участия Hero в шоудауне (вероятно, Hero сфолдил или выиграл без вскрытия).\n"
    await update.message.reply_text(text, parse_mode='Markdown')

async def explain_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    last = context.user_data.get('last_hand_info')
    parsed = context.user_data.get('last_parsed_hand')
    if not last or not parsed:
        await update.message.reply_text("❌ Нет данных для объяснения. Сначала отправьте раздачу.")
        return

    reasons_pre = []
    if last['is_pair']:
        if last['high_card'] in ['A','K','Q','J','T']:
            reasons_pre.append(f"🔹 Сильная карманная пара ({last['high_card']}{last['high_card']})")
        else:
            reasons_pre.append(f"🔹 Слабая пара ({last['high_card']}{last['high_card']})")
    else:
        if last['suited'] and last['high_card'] in ['A','K','Q','J']:
            reasons_pre.append(f"🔹 Хорошая одномастная рука ({last['high_card']}{last['low_card']} одномастные)")
        else:
            reasons_pre.append(f"🔹 Слабая рука ({last['high_card']}{last['low_card']}, не пара, не одномастные)")
    if last['position'] in ['EP','MP1','MP2']:
        reasons_pre.append(f"🔹 Ранняя позиция ({last['position']}) – рискованно")
    elif last['position'] in ['MP3','CO']:
        reasons_pre.append(f"🔹 Средняя позиция ({last['position']})")
    else:
        reasons_pre.append(f"🔹 Поздняя позиция ({last['position']}) – преимущество")
    if last['opponents'] >= 4:
        reasons_pre.append(f"🔹 Много оппонентов ({last['opponents']}) – нужна сильная рука")
    if last['stack_bb'] < 20:
        reasons_pre.append(f"🔹 Короткий стек ({last['stack_bb']:.0f} BB) – либо оллин, либо фолд")
    conclusion_pre = {
        'fold': '🎯 Модель советует **СБРОСИТЬ**',
        'call': '🎯 Модель советует **УРАВНЯТЬ**',
        'raise': '🎯 Модель советует **ПОВЫСИТЬ**',
        'check': '🎯 Модель советует **ЧЕКНУТЬ**'
    }.get(last['action'], '')
    text = f"🔍 *Объяснение на префлопе:* {last['action'].upper()} (уверенность {last['confidence']:.0%})\n\n"
    text += "\n".join(reasons_pre) + f"\n\n{conclusion_pre}\n\n"

    for street in ['flop', 'turn', 'river']:
        action_key = f'{street}_action'
        cards_key = f'{street}_cards'
        action = parsed.get(action_key)
        cards = parsed.get(cards_key)
        if action:
            pred_action, conf, _ = get_postflop_prediction(parsed, street)
            if pred_action:
                emoji = {'flop': '♣️', 'turn': '♦️', 'river': '♥️'}[street]
                if cards:
                    texture = analyze_board_texture(cards)
                    texture_str = f"\n   📌 *Анализ доски:* {texture}"
                    pot = parsed.get(f'{street}_pot', 0)
                    bet = parsed.get(f'{street}_bet', 0)
                    draws_analysis = calculate_draws_and_odds(last['cards'], cards, street, pot, bet)
                    if draws_analysis:
                        texture_str += f"\n   🧩 *Дро и шансы:* {draws_analysis}"
                else:
                    texture_str = ""
                text += f"{emoji} *{street.upper()}* (реальное действие: {action.upper()})\n"
                text += f"   🎯 *Рекомендация модели:* {pred_action.upper()} (уверенность {conf:.0%}){texture_str}\n"
                if pred_action == 'fold':
                    text += "   ✨ *Почему:* Рука слабая, слишком много оппонентов или опасная доска.\n"
                elif pred_action == 'call':
                    text += "   ✨ *Почему:* Рука имеет потенциал, но не настолько сильная для рейза.\n"
                elif pred_action in ('bet', 'raise'):
                    text += "   ✨ *Почему:* У вас сильная рука или хорошая возможность для блефа.\n"
                elif pred_action == 'check':
                    text += "   ✨ *Почему:* Нет необходимости ставить, можно посмотреть бесплатную карту.\n"
                if action == pred_action:
                    text += f"   ✅ *Вердикт:* Ваше действие совпало с рекомендацией. Молодец!\n"
                else:
                    text += f"   ⚠️ *Вердикт:* Модель советовала {pred_action.upper()}. Подумайте, почему возможно было сыграть иначе.\n"
                text += "\n"
    await update.message.reply_text(text, parse_mode='Markdown')

async def postflop_command(update: Update, context: ContextTypes.DEFAULT_TYPE, street: str):
    parsed = context.user_data.get('last_parsed_hand')
    if not parsed:
        await update.message.reply_text("❌ Нет данных о последней руке. Сначала отправьте раздачу.")
        return
    cards = parsed.get(f'{street}_cards')
    action = parsed.get(f'{street}_action')
    cards_text = f"🃏 Карты {street.upper()}: {cards}" if cards else f"⚠️ Карты {street.upper()} не найдены в раздаче."
    if action:
        await update.message.reply_text(f"{cards_text}\nℹ️ В этой раздаче на {street.upper()} вы уже совершили действие: {action.upper()}.")
        return
    pred_action, confidence, _ = get_postflop_prediction(parsed, street)
    if pred_action is None:
        await update.message.reply_text(f"{cards_text}\n❌ Модель для {street.upper()} не загружена или недостаточно данных.")
        return
    emoji = {'fold':'🤚', 'call':'📞', 'bet':'💰', 'raise':'📈', 'check':'✅'}
    reply = f"{cards_text}\n🎯 *Предсказание на {street.upper()}:* {pred_action.upper()} {emoji.get(pred_action, '')} (уверенность {confidence:.1%})"
    if cards and parsed.get('hero_hole_cards'):
        pot = parsed.get(f'{street}_pot', 0)
        bet = parsed.get(f'{street}_bet', 0)
        draws = calculate_draws_and_odds(parsed['hero_hole_cards'], cards, street, pot, bet)
        if draws:
            reply += f"\n\n🧩 *Анализ:* {draws}"
    await update.message.reply_text(reply, parse_mode='Markdown')

async def flop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await postflop_command(update, context, 'flop')

async def turn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await postflop_command(update, context, 'turn')

async def river(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await postflop_command(update, context, 'river')

async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['awaiting_feedback'] = True
    await update.message.reply_text("📝 Напишите ваш отзыв или пожелание одним сообщением. Администратор получит его.")

async def terms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "⚖️ *Условия использования*\n\n"
        "1️⃣ Предсказания носят развлекательный характер и не гарантируют выигрыш.\n"
        "2️⃣ Бот не сохраняет тексты раздач дольше, чем необходимо для сбора обратной связи.\n"
        "3️⃣ Ваши оценки (✅/❌) анонимно используются для дообучения модели.\n\n"
        "✅ Продолжая использовать бота, вы соглашаетесь с этими условиями."
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def reload_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("⛔ Доступ запрещён.")
        return
    load_model_files()
    load_postflop_models()
    await update.message.reply_text("✅ Все модели перезагружены из файлов.")
    logger.info("Модели перезагружены админом")

async def retrain_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("⛔ Доступ запрещён.")
        return
    await update.message.reply_text("🔄 Запускаю переобучение модели (может занять до минуты)...")
    try:
        result = subprocess.run([sys.executable, "auto_retrain.py"], capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            await update.message.reply_text("✅ Переобучение завершено. Перезагружаю модели...")
            load_model_files()
            load_postflop_models()
            await update.message.reply_text("✅ Модели успешно перезагружены.")
        else:
            await update.message.reply_text(f"❌ Ошибка переобучения:\n{result.stderr[:1000]}")
    except subprocess.TimeoutExpired:
        await update.message.reply_text("❌ Превышено время ожидания (120 секунд).")
    except Exception as e:
        await update.message.reply_text(f"❌ Исключение: {e}")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if model is None:
        await update.message.reply_text("❌ Модель не загружена.")
        return
    doc = update.message.document
    if not doc.file_name.endswith('.txt'):
        await update.message.reply_text("Пожалуйста, отправьте файл .txt")
        return
    file = await doc.get_file()
    content_bytes = await file.download_as_bytearray()
    for enc in ['utf-8', 'cp1251', 'latin1']:
        try:
            text = content_bytes.decode(enc)
            break
        except:
            continue
    else:
        await update.message.reply_text("Не удалось прочитать файл.")
        return
    await predict(update, text, context)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if model is None:
        await update.message.reply_text("❌ Модель не загружена.")
        return
    text = update.message.text
    if text.startswith('/'):
        return
    if context.user_data.get('awaiting_feedback'):
        context.user_data['awaiting_feedback'] = False
        user = update.effective_user
        feedback_text = f"📝 *Новый отзыв*\nОт: @{user.username or user.first_name} (ID: {user.id})\n\n{text}"
        if ADMIN_ID == 0:
            await update.message.reply_text("❌ Администратор не настроен. Отзыв не может быть отправлен.")
            return
        try:
            await context.bot.send_message(chat_id=ADMIN_ID, text=feedback_text, parse_mode='Markdown')
            await update.message.reply_text("🙏 Спасибо за ваш отзыв! Он передан администратору.")
        except Exception as e:
            logger.error(f"Не удалось отправить отзыв админу: {e}")
            await update.message.reply_text("❌ Не удалось отправить отзыв. Попробуйте позже.")
        return
    await predict(update, text, context)

async def predict(update: Update, text: str, context: ContextTypes.DEFAULT_TYPE):
    if 'Poker Hand #' in text:
        first = re.search(r'(Poker Hand #.*?)(?=Poker Hand #|$)', text, re.DOTALL)
        if first:
            text = first.group(1)
    parsed = parse_hand_advanced(text)
    if not parsed or not parsed['hero_hole_cards']:
        await update.message.reply_text("❌ Не удалось распознать раздачу. Убедитесь в формате.")
        return
    context.user_data['last_parsed_hand'] = parsed

    card_feats = hand_features(parsed['hero_hole_cards'])
    pos_num = POS_ORDER.get(parsed['hero_position'], 7)
    features = [
        pos_num, parsed['hero_stack_pre_bb'], parsed['preflop_opponents'],
        card_feats['is_pair'], card_feats['suited'],
        card_feats['high_card_rank'], card_feats['low_card_rank'],
        card_feats['gap'], card_feats['hand_group']
    ]
    X = np.array([features])
    pred_enc = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    confidence = np.max(probs)
    action = encoder.inverse_transform([pred_enc])[0]
    emoji = {'fold':'🤚', 'call':'📞', 'raise':'📈', 'check':'✅'}
    reply = f"🎯 *Предсказание:* {action.upper()} {emoji.get(action,'')} (уверенность {confidence:.1%})"

    pred_id = save_prediction(update.effective_user.id, text[:1000], features, action)
    keyboard = [[InlineKeyboardButton("✅ Верно", callback_data=f"feedback_{pred_id}_yes"),
                 InlineKeyboardButton("❌ Неверно", callback_data=f"feedback_{pred_id}_no")]]
    await update.message.reply_text(reply, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))

    high_char = rank_to_letter.get(card_feats['high_card_rank'], str(card_feats['high_card_rank']))
    low_char = rank_to_letter.get(card_feats['low_card_rank'], str(card_feats['low_card_rank']))
    context.user_data['last_hand_info'] = {
        'cards': parsed['hero_hole_cards'],
        'position': parsed['hero_position'],
        'stack_bb': parsed['hero_stack_pre_bb'],
        'opponents': parsed['preflop_opponents'],
        'is_pair': card_feats['is_pair'],
        'suited': card_feats['suited'],
        'high_card': high_char,
        'low_card': low_char,
        'high_rank': card_feats['high_card_rank'],
        'low_rank': card_feats['low_card_rank'],
        'gap': card_feats['gap'],
        'hand_group': card_feats['hand_group'],
        'action': action,
        'confidence': confidence,
        'flop_cards': parsed['flop_cards'],
        'flop_action': parsed['flop_action'],
        'flop_opponents': parsed['flop_opponents'],
        'flop_cbet': parsed['flop_cbet'],
        'turn_cards': parsed['turn_cards'],
        'turn_action': parsed['turn_action'],
        'turn_opponents': parsed['turn_opponents'],
        'river_cards': parsed['river_cards'],
        'river_action': parsed['river_action'],
        'river_opponents': parsed['river_opponents']
    }
    context.user_data['predictions'] = context.user_data.get('predictions', 0) + 1

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data.startswith('feedback_'):
        parts = query.data.split('_')
        if len(parts) == 3:
            pred_id = parts[1]
            feedback = 'yes' if parts[2] == 'yes' else 'no'
            update_feedback(pred_id, feedback)
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("🙏 Спасибо за обратную связь!")

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(CommandHandler("profile", profile))
    app.add_handler(CommandHandler("analysis", analysis))
    app.add_handler(CommandHandler("explain", explain_command))
    app.add_handler(CommandHandler("flop", flop))
    app.add_handler(CommandHandler("turn", turn))
    app.add_handler(CommandHandler("river", river))
    app.add_handler(CommandHandler("feedback", feedback_command))
    app.add_handler(CommandHandler("terms", terms))
    app.add_handler(CommandHandler("reload", reload_model))
    app.add_handler(CommandHandler("retrain_now", retrain_now))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(callback_handler))
    logger.info("Бот запущен")
    app.run_polling()

if __name__ == "__main__":
    main()
