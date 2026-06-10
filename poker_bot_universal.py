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
        'turn_action': None,
        'turn_opponents': 0,
        'turn_cards': None,
        'river_action': None,
        'river_opponents': 0,
        'river_cards': None,
        'board_cards': None,
        'showdown_hero_hand': None,
        'showdown_hero_rank': None,
        'showdown_villain_hand': None,
        'showdown_villain_rank': None,
        'hero_won': None,
        'hero_win_amount': 0.0,
        'flop_pot': 0.0,
        'flop_call_amount': 0.0,
        'turn_pot': 0.0,
        'turn_call_amount': 0.0,
        'river_pot': 0.0,
        'river_call_amount': 0.0,
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

    # Карты стола — работает с *** FLOP *** (GG/Party) и просто FLOP (PokerStars)
    flop_match = re.search(r'(?:\*+\s*)?FLOP(?:\s*\*+)?\s*\[([^\]]+)\]', content, re.IGNORECASE)
    if flop_match:
        result['flop_cards'] = flop_match.group(1).strip()

    # Тёрн: ищем второй блок скобок на строке с TURN [flop_cards] [turn_card]
    turn_match = re.search(r'(?:\*+\s*)?TURN(?:\s*\*+)?\s*\[[^\]]+\]\s*\[([^\]]+)\]', content, re.IGNORECASE)
    if turn_match:
        result['turn_cards'] = turn_match.group(1).strip()
    else:
        # fallback: строка "TURN [2s]" без предыдущих карт
        turn_match2 = re.search(r'(?:\*+\s*)?TURN(?:\s*\*+)?\s*\[([^\]]+)\]', content, re.IGNORECASE)
        if turn_match2:
            cards = turn_match2.group(1).strip().split()
            result['turn_cards'] = cards[-1] if cards else None

    # Ривер: аналогично
    river_match = re.search(r'(?:\*+\s*)?RIVER(?:\s*\*+)?\s*\[[^\]]+\]\s*\[([^\]]+)\]', content, re.IGNORECASE)
    if river_match:
        result['river_cards'] = river_match.group(1).strip()
    else:
        river_match2 = re.search(r'(?:\*+\s*)?RIVER(?:\s*\*+)?\s*\[([^\]]+)\]', content, re.IGNORECASE)
        if river_match2:
            cards = river_match2.group(1).strip().split()
            result['river_cards'] = cards[-1] if cards else None

    # Полная доска для анализа
    board_cards = []
    if result['flop_cards']:
        board_cards.extend(result['flop_cards'].split())
    if result['turn_cards']:
        board_cards.append(result['turn_cards'])
    if result['river_cards']:
        board_cards.append(result['river_cards'])
    result['board_cards'] = " ".join(board_cards) if board_cards else None

    # Функция для извлечения суммы из строки действия
    def get_bet_amount(line):
        m = re.search(r'posts (?:small blind|big blind|ante)\s+\$([\d\.]+)', line, re.IGNORECASE)
        if m:
            return float(m.group(1))
        m = re.search(r'calls?\s+\$([\d\.]+)', line, re.IGNORECASE)
        if m:
            return float(m.group(1))
        m = re.search(r'bets\s+\$([\d\.]+)', line, re.IGNORECASE)
        if m:
            return float(m.group(1))
        # "raises $X to $Y" — в банк идёт итоговая сумма колла (to $Y)
        m = re.search(r'raises\s+\$([\d\.]+)\s+to\s+\$([\d\.]+)', line, re.IGNORECASE)
        if m:
            return float(m.group(2))
        m = re.search(r'raises\s+\$([\d\.]+)', line, re.IGNORECASE)
        if m:
            return float(m.group(1))
        return 0.0

    # Функция для получения суммы колла Hero на улице
    def get_hero_call_amount(block_text):
        """Возвращает сумму, которую Hero должен заколлить (последняя ставка/рейз оппонента до действия Hero)"""
        lines = block_text.split('\n')
        last_bet = 0.0
        for line in lines:
            if 'Hero' in line:
                break
            m = re.search(r'bets\s+\$([\d\.]+)', line, re.IGNORECASE)
            if m:
                last_bet = float(m.group(1))
            m = re.search(r'raises\s+\$([\d\.]+)\s+to\s+\$([\d\.]+)', line, re.IGNORECASE)
            if m:
                last_bet = float(m.group(2))
            m = re.search(r'raises\s+\$([\d\.]+)', line, re.IGNORECASE)
            if m:
                last_bet = float(m.group(1))
        return last_bet

    # Разбивка на блоки — работает с *** и без ***
    pre_block = re.search(r'(?:HOLE CARDS.*?)(?=(?:\*+\s*)?FLOP|\Z)', content, re.DOTALL | re.IGNORECASE)
    pre_text = pre_block.group(0) if pre_block else ""
    flop_block = re.search(r'(?:(?:\*+\s*)?FLOP.*?)(?=(?:\*+\s*)?TURN|\Z)', content, re.DOTALL | re.IGNORECASE)
    flop_text = flop_block.group(0) if flop_block else ""
    turn_block = re.search(r'(?:(?:\*+\s*)?TURN.*?)(?=(?:\*+\s*)?RIVER|\Z)', content, re.DOTALL | re.IGNORECASE)
    turn_text = turn_block.group(0) if turn_block else ""
    river_block = re.search(r'(?:(?:\*+\s*)?RIVER.*?)(?=(?:\*+\s*)?SHOW DOWN|\Z)', content, re.DOTALL | re.IGNORECASE)
    river_text = river_block.group(0) if river_block else ""

    # Расчёт банка (накопительно по улицам)
    pot = 0.0

    # Анте и блайнды — идут ДО блока HOLE CARDS, считаем отдельно
    antes_block = re.search(r'^(.*?)(?=\*+\s*HOLE CARDS)', content, re.DOTALL | re.IGNORECASE)
    if antes_block:
        for line in antes_block.group(1).split('\n'):
            pot += get_bet_amount(line)

    # Префлоп (действия после HOLE CARDS: коллы, рейзы)
    for line in pre_text.split('\n'):
        pot += get_bet_amount(line)

    pre_action_match = re.search(r'Hero:\s*(folds|checks|calls|bets|raises)', pre_text, re.IGNORECASE)
    if pre_action_match:
        result['preflop_action'] = pre_action_match.group(1).lower()
    pre_players = set(re.findall(r'([A-Za-z0-9_]+):\s*(?:folds|raises|calls|bets|checks)', pre_text))
    pre_players.discard('Hero')
    result['preflop_opponents'] = len(pre_players)

    # Флоп
    result['flop_pot'] = pot  # банк ДО действий на флопе
    flop_call_amount = get_hero_call_amount(flop_text)
    result['flop_call_amount'] = flop_call_amount
    for line in flop_text.split('\n'):
        pot += get_bet_amount(line)

    flop_action_match = re.search(r'Hero:\s*(folds|checks|calls|bets|raises)', flop_text, re.IGNORECASE)
    if flop_action_match:
        result['flop_action'] = flop_action_match.group(1).lower()
    flop_players = set(re.findall(r'([A-Za-z0-9_]+):\s*(?:folds|raises|calls|bets|checks)', flop_text))
    flop_players.discard('Hero')
    result['flop_opponents'] = len(flop_players)
    if result['flop_action'] in ['bets', 'raises', 'bet', 'raise']:
        result['flop_cbet'] = 1

    # Тёрн
    result['turn_pot'] = pot  # банк ДО действий на тёрне
    turn_call_amount = get_hero_call_amount(turn_text)
    result['turn_call_amount'] = turn_call_amount
    for line in turn_text.split('\n'):
        pot += get_bet_amount(line)

    turn_action_match = re.search(r'Hero:\s*(folds|checks|calls|bets|raises)', turn_text, re.IGNORECASE)
    if turn_action_match:
        result['turn_action'] = turn_action_match.group(1).lower()
    turn_players = set(re.findall(r'([A-Za-z0-9_]+):\s*(?:folds|raises|calls|bets|checks)', turn_text))
    turn_players.discard('Hero')
    result['turn_opponents'] = len(turn_players)

    # Ривер
    result['river_pot'] = pot  # банк ДО действий на ривере
    river_call_amount = get_hero_call_amount(river_text)
    result['river_call_amount'] = river_call_amount
    for line in river_text.split('\n'):
        pot += get_bet_amount(line)

    river_action_match = re.search(r'Hero:\s*(folds|checks|calls|bets|raises)', river_text, re.IGNORECASE)
    if river_action_match:
        result['river_action'] = river_action_match.group(1).lower()
    river_players = set(re.findall(r'([A-Za-z0-9_]+):\s*(?:folds|raises|calls|bets|checks)', river_text))
    river_players.discard('Hero')
    result['river_opponents'] = len(river_players)

    # Шоудаун
    hero_show = re.search(r'Hero[:\s]+shows?ed?\s+\[([^]]+)\]\s*\(([^)]+)\)', content, re.IGNORECASE)
    if not hero_show:
        hero_show = re.search(r'Hero[:\s]+shows?ed?\s+\[([^]]+)\][:\s]+and (?:won|lost) with\s+(.+)', content, re.IGNORECASE)
    if hero_show:
        result['showdown_hero_hand'] = hero_show.group(1).strip()
        result['showdown_hero_rank'] = hero_show.group(2).strip()
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

# ---------- Расчёт шансов банка (Pot Odds) ----------
def calculate_pot_odds(pot_size, call_amount):
    """
    Рассчитывает pot odds и минимально необходимый equity для прибыльного колла.
    
    pot_size   — банк ДО колла
    call_amount — сумма, которую нужно заколлить
    
    Возвращает строку с готовым анализом.
    """
    if call_amount <= 0 or pot_size <= 0:
        return ""

    # Pot odds как соотношение X:1
    odds_ratio = pot_size / call_amount

    # Минимальный equity для безубыточного колла
    # equity_needed = call / (pot + call)
    equity_needed = call_amount / (pot_size + call_amount) * 100

    # Оценка выгодности
    if equity_needed <= 20:
        verdict = "🟢 Очень выгодный колл — нужен минимальный equity"
    elif equity_needed <= 33:
        verdict = "🟢 Выгодный колл — хорошие шансы банка"
    elif equity_needed <= 40:
        verdict = "🟡 Умеренный колл — нужна приличная рука или дро"
    elif equity_needed <= 50:
        verdict = "🟠 Сомнительный колл — нужно крепкое дро или топ-пара"
    else:
        verdict = "🔴 Невыгодный колл — нужен очень сильный equity"

    return (
        f"💰 *Шансы банка:* ${pot_size:.2f} в банке / ${call_amount:.2f} колл = {odds_ratio:.1f}:1\n"
        f"   📊 Нужен equity ≥ {equity_needed:.0f}% для прибыльного колла\n"
        f"   {verdict}"
    )

# ---------- Анализ дро и эквити (ИСПРАВЛЕНО: принимает полную доску) ----------
def calculate_draws_and_odds(hero_cards, board_cards_str, street):
    """
    Анализирует дро Hero с учётом ПОЛНОЙ текущей доски.
    board_cards_str — строка со всеми картами доски на данной улице.
    """
    if not hero_cards or not board_cards_str:
        return ""

    hero_list = hero_cards.split()
    board_list = board_cards_str.split()
    all_cards = hero_list + board_list

    hero_suits = [c[1] for c in hero_list]
    hero_ranks = [RANK_ORDER.get(c[0], 0) for c in hero_list]
    board_ranks = [RANK_ORDER.get(c[0], 0) for c in board_list]
    all_ranks = [RANK_ORDER.get(c[0], 0) for c in all_cards]

    draws = []
    total_outs = 0

    # ── Флеш-дро ──────────────────────────────────────────────────────────────
    if hero_suits[0] == hero_suits[1]:
        suit = hero_suits[0]
        suited_on_board = sum(1 for c in board_list if c[1] == suit)
        total_suited = suited_on_board + 2  # 2 карты в руке
        if total_suited >= 5:
            draws.append("💧 Флеш уже собран! (5+ карт одной масти)")
        elif total_suited == 4:
            outs = 9
            draws.append(f"💧 Флеш-дро ({outs} аутов)")
            total_outs += outs
        elif total_suited == 3:
            draws.append("💧 Бэкдор-флеш (нужны две карты одной масти)")

    # ── Стрит-дро ─────────────────────────────────────────────────────────────
    sorted_ranks = sorted(set(all_ranks))
    oesd_found = False
    gutshot_found = False

    for i in range(len(sorted_ranks)):
        for j in range(i + 1, len(sorted_ranks)):
            span = sorted_ranks[j] - sorted_ranks[i]
            count = j - i + 1
            if span <= 4 and count >= 4:
                # Открытый стрит (OESD) — 4 последовательные карты без пропусков
                if span == 3 and not oesd_found:
                    draws.append("📏 OESD — Открытый стрит-дро (8 аутов)")
                    total_outs += 8
                    oesd_found = True
                # Гатшот — 4 карты с одним пропуском
                elif span == 4 and not gutshot_found and not oesd_found:
                    draws.append("📏 Гатшот — стрит-дро с одним пропуском (4 аута)")
                    total_outs += 4
                    gutshot_found = True

    # ── Пара/трипс ────────────────────────────────────────────────────────────
    for hr in hero_ranks:
        if hr in board_ranks:
            board_count = board_ranks.count(hr)
            if board_count == 1:
                draws.append("🎯 Пара с доской — возможен трипс (2 аута)")
                total_outs += 2
                break
            elif board_count == 2:
                draws.append("🎯 Трипс — возможен каре (1 аут)")
                total_outs += 1
                break

    # ── Оверкарты ─────────────────────────────────────────────────────────────
    if not draws:
        max_board = max(board_ranks) if board_ranks else 0
        overcards = [r for r in hero_ranks if r > max_board]
        if len(overcards) == 2:
            draws.append(f"🔸 Две оверкарты (6 аутов для топ-пары)")
            total_outs += 6
        elif len(overcards) == 1:
            draws.append(f"🔸 Одна оверкарта (3 аута)")
            total_outs += 3

    # ── Эквити по правилу 2/4 ─────────────────────────────────────────────────
    if total_outs > 0:
        if street == 'flop':
            # На флопе ещё 2 карты — умножаем на 4
            equity = min(total_outs * 4, 100)
            draws.append(f"📈 Equity ≈ {equity}% (правило ×4, флоп→ривер)")
        elif street == 'turn':
            # На тёрне ещё 1 карта — умножаем на 2
            equity = min(total_outs * 2, 100)
            draws.append(f"📈 Equity ≈ {equity}% (правило ×2, тёрн→ривер)")
        else:
            draws.append("🔹 Ривер — все карты открыты, equity рассчитывать нечего")
    else:
        draws.append("🔹 Нет значимых дро на этой улице")

    return "\n".join(draws)

# ---------- Анализ шансов банка vs equity дро ----------
def pot_odds_vs_equity(pot_size, call_amount, hero_cards, board_cards_str, street):
    """
    Сравнивает pot odds с equity дро — стоит ли коллить?
    Возвращает строку с итоговым вердиктом.
    """
    if call_amount <= 0 or pot_size <= 0 or not hero_cards or not board_cards_str:
        return ""

    equity_needed = call_amount / (pot_size + call_amount) * 100

    # Грубо оцениваем equity из аутов
    hero_list = hero_cards.split()
    board_list = board_cards_str.split()
    all_cards = hero_list + board_list
    hero_suits = [c[1] for c in hero_list]
    hero_ranks = [RANK_ORDER.get(c[0], 0) for c in hero_list]
    board_ranks = [RANK_ORDER.get(c[0], 0) for c in board_list]
    all_ranks = [RANK_ORDER.get(c[0], 0) for c in all_cards]

    total_outs = 0
    if hero_suits[0] == hero_suits[1]:
        suit = hero_suits[0]
        suited_on_board = sum(1 for c in board_list if c[1] == suit)
        if suited_on_board + 2 == 4:
            total_outs += 9

    sorted_ranks = sorted(set(all_ranks))
    for i in range(len(sorted_ranks)):
        for j in range(i + 1, len(sorted_ranks)):
            span = sorted_ranks[j] - sorted_ranks[i]
            count = j - i + 1
            if span <= 4 and count >= 4:
                if span == 3:
                    total_outs += 8
                    break
                elif span == 4:
                    total_outs += 4
                    break

    for hr in hero_ranks:
        if hr in board_ranks:
            total_outs += 2
            break

    if total_outs == 0:
        max_board = max(board_ranks) if board_ranks else 0
        overcards = [r for r in hero_ranks if r > max_board]
        total_outs += len(overcards) * 3

    if street == 'flop':
        estimated_equity = min(total_outs * 4, 100)
    elif street == 'turn':
        estimated_equity = min(total_outs * 2, 100)
    else:
        estimated_equity = 0

    if estimated_equity == 0 or total_outs == 0:
        return ""

    margin = estimated_equity - equity_needed
    if margin >= 10:
        verdict = f"✅ *Колл выгоден*: equity дро ≈{estimated_equity}% > нужно {equity_needed:.0f}% (+{margin:.0f}%)"
    elif margin >= 0:
        verdict = f"🟡 *Колл на грани*: equity дро ≈{estimated_equity}% ≈ нужно {equity_needed:.0f}%"
    else:
        verdict = f"❌ *Колл невыгоден по дро*: equity ≈{estimated_equity}% < нужно {equity_needed:.0f}% ({margin:.0f}%)"

    return verdict

# ---------- Текстура доски (ИСПРАВЛЕНО: принимает список карт) ----------
def analyze_board_texture(board_cards_str):
    """Анализирует текстуру доски. board_cards_str — строка со всеми картами доски."""
    if not board_cards_str:
        return ""
    cards = board_cards_str.strip().split()
    if not cards:
        return ""

    suits = [c[1] for c in cards]
    ranks = [c[0] for c in cards]
    rank_values = [RANK_ORDER.get(r, 0) for r in ranks]

    is_monotone = len(set(suits)) == 1
    is_paired = len(set(ranks)) < len(ranks)
    suit_counts = {s: suits.count(s) for s in set(suits)}
    flush_draw = any(count >= 3 for count in suit_counts.values())
    flush_draw_2 = any(count >= 2 for count in suit_counts.values()) and not flush_draw
    sorted_ranks = sorted(rank_values)
    straight_possible = False
    if len(sorted_ranks) >= 3 and max(sorted_ranks) - min(sorted_ranks) <= 4:
        straight_possible = True

    parts = []
    if is_monotone:
        parts.append("💧 Монотонная доска — высокий риск флеша")
    elif flush_draw:
        parts.append("💧 Три карты одной масти — флеш уже возможен")
    elif flush_draw_2:
        parts.append("💧 Флеш-дро на доске (2 карты одной масти)")

    if is_paired:
        parts.append("🪵 Парная доска — возможен фулл-хаус или трипс")

    if straight_possible:
        parts.append("📏 Координированная доска — возможен стрит")

    if not parts:
        parts.append("🍂 Сухая доска — меньше дро у оппонентов")

    return " | ".join(parts)

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
        m = flop_model
        enc = flop_encoder
        expected_cols = flop_model.feature_names_in_
    elif street == 'turn':
        m = turn_model
        enc = turn_encoder
        expected_cols = turn_model.feature_names_in_
    else:
        m = river_model
        enc = river_encoder
        expected_cols = river_model.feature_names_in_
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[expected_cols].fillna(0).values
    pred_enc = m.predict(X)[0]
    probs = m.predict_proba(X)[0]
    confidence = np.max(probs)
    action = enc.inverse_transform([pred_enc])[0]
    return action, confidence, probs

# ---------- Нормализация действий для сравнения ----------
def normalize_action(action):
    """Приводит варианты написания к единому виду."""
    if not action:
        return ''
    a = action.lower().strip()
    if a in ('bets', 'bet'):
        return 'bet'
    if a in ('raises', 'raise'):
        return 'raise'
    if a in ('calls', 'call'):
        return 'call'
    if a in ('folds', 'fold'):
        return 'fold'
    if a in ('checks', 'check'):
        return 'check'
    return a

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
        "/analysis – подробный разбор с шансами банка, анализом дро и сравнением equity\n"
        "/explain – понятное объяснение, почему бот дал такой совет\n"
        "/flop, /turn, /river – предсказания на соответствующих улицах"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 *Poker Oracle Bot v4.0* (расчёт pot odds на каждой улице)\n\n"
        "🧠 *Модели:* Random Forest для префлопа, флопа, тёрна, ривера.\n"
        "📊 *Признаки:* позиция, сила руки, стек, оппоненты, предыдущие действия.\n"
        "♠️ *Поддерживаемые румы:* PokerStars, GG Poker, PartyPoker.\n"
        "📈 *Функции:*\n"
        "• Предсказание действий на всех улицах\n"
        "• Расчёт шансов банка (pot odds) и нужного equity\n"
        "• Сравнение pot odds vs equity ваших дро\n"
        "• Анализ текстуры доски (вся доска, а не одна карта)\n"
        "• Обратная связь и переобучение модели\n\n"
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
        f"📊 *Признаки руки:*\n"
        f"• Пара: {'да' if last['is_pair'] else 'нет'}\n"
        f"• Одномастные: {'да' if last['suited'] else 'нет'}\n"
        f"• Старшая карта: {last['high_card']}\n"
        f"• Младшая карта: {last['low_card']}\n"
        f"• Разрыв (gap): {last['gap']}\n"
        f"• Группа руки: {last['hand_group']}\n\n"
        f"🤖 *Предсказание на префлопе:* *{last['action'].upper()}* (уверенность {last['confidence']:.1%})\n"
    )

    user_preflop = normalize_action(parsed.get('preflop_action'))
    model_preflop = normalize_action(last['action'])
    if user_preflop:
        if user_preflop == model_preflop:
            text += f"\n✅ *Префлоп:* Вы {user_preflop.upper()} — совпадает с рекомендацией модели.\n"
        else:
            text += f"\n⚠️ *Префлоп:* Вы {user_preflop.upper()}, модель рекомендовала {model_preflop.upper()} (уверенность {last['confidence']:.1%}).\n"

    # ── Постфлоп-улицы ────────────────────────────────────────────────────────
    street_emojis = {'flop': '♣️', 'turn': '♦️', 'river': '♥️'}
    street_names = {'flop': 'Флоп', 'turn': 'Тёрн', 'river': 'Ривер'}

    # Накопленная доска для корректного анализа
    board_so_far = []

    for street in ['flop', 'turn', 'river']:
        street_card = parsed.get(f'{street}_cards')
        if not street_card:
            continue

        # Добавляем карту(ы) улицы в накопленную доску
        if street == 'flop':
            board_so_far.extend(street_card.split())
        else:
            board_so_far.append(street_card)
        current_board = " ".join(board_so_far)

        emoji = street_emojis[street]
        name = street_names[street]
        text += f"\n{emoji} *{name}:* {current_board}\n"

        # Текстура на основе ПОЛНОЙ текущей доски
        texture = analyze_board_texture(current_board)
        if texture:
            text += f"   📋 *Текстура:* {texture}\n"

        # Анализ дро на основе полной доски
        draws_analysis = calculate_draws_and_odds(last['cards'], current_board, street)
        if draws_analysis:
            text += f"   🧩 *Дро:* {draws_analysis}\n"

        # Шансы банка
        pot_before = parsed.get(f'{street}_pot', 0)
        call_amount = parsed.get(f'{street}_call_amount', 0)
        if call_amount > 0 and pot_before > 0:
            pot_odds_str = calculate_pot_odds(pot_before, call_amount)
            if pot_odds_str:
                text += f"   {pot_odds_str}\n"
            # Сравнение pot odds vs equity дро
            verdict = pot_odds_vs_equity(pot_before, call_amount, last['cards'], current_board, street)
            if verdict:
                text += f"   {verdict}\n"

        # Действие пользователя и оценка модели
        user_action_raw = parsed.get(f'{street}_action')
        if user_action_raw:
            user_action = normalize_action(user_action_raw)
            opponents_count = parsed.get(f'{street}_opponents', 0)
            cbet_str = " (конт-бет ✅)" if street == 'flop' and parsed.get('flop_cbet') else ""
            text += f"   👤 *Ваше действие:* {user_action.upper()}, оппонентов — {opponents_count}{cbet_str}\n"

            pred_action, pred_conf, _ = get_postflop_prediction(parsed, street)
            if pred_action:
                pred_norm = normalize_action(pred_action)
                if pred_norm == user_action:
                    text += f"   ✅ *Оценка:* Совпадает с рекомендацией модели ({pred_norm.upper()}, уверенность {pred_conf:.1%}).\n"
                else:
                    text += f"   ⚠️ *Оценка:* Модель рекомендовала {pred_norm.upper()} (уверенность {pred_conf:.1%}), вы сделали {user_action.upper()}.\n"

    # ── Итог раздачи ──────────────────────────────────────────────────────────
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
        text += "⚠️ *Результат:* Раздача завершена без участия Hero в шоудауне.\n"

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
            reasons_pre.append(f"🔹 Хорошая одномастная рука ({last['high_card']}{last['low_card']} сьютед)")
        else:
            reasons_pre.append(f"🔹 Рука: {last['high_card']}{last['low_card']} ({'сьютед' if last['suited'] else 'оффсьют'})")
    if last['position'] in ['EP','MP1','MP2']:
        reasons_pre.append(f"🔹 Ранняя позиция ({last['position']}) — рискованно")
    elif last['position'] in ['MP3','CO']:
        reasons_pre.append(f"🔹 Средняя позиция ({last['position']})")
    else:
        reasons_pre.append(f"🔹 Поздняя позиция ({last['position']}) — преимущество")
    if last['opponents'] >= 4:
        reasons_pre.append(f"🔹 Много оппонентов ({last['opponents']}) — нужна сильная рука")
    if last['stack_bb'] < 20:
        reasons_pre.append(f"🔹 Короткий стек ({last['stack_bb']:.0f} BB) — либо оллин, либо фолд")
    conclusion_pre = {
        'fold': '🎯 Модель советует *СБРОСИТЬ*',
        'call': '🎯 Модель советует *УРАВНЯТЬ*',
        'raise': '🎯 Модель советует *ПОВЫСИТЬ*',
        'check': '🎯 Модель советует *ЧЕКНУТЬ*'
    }.get(last['action'], '')
    text = f"🔍 *Объяснение на префлопе:* {last['action'].upper()} (уверенность {last['confidence']:.0%})\n\n"
    text += "\n".join(reasons_pre) + f"\n\n{conclusion_pre}\n\n"

    board_so_far = []
    for street in ['flop', 'turn', 'river']:
        action_key = f'{street}_action'
        cards_key = f'{street}_cards'
        action_raw = parsed.get(action_key)
        cards = parsed.get(cards_key)
        if not action_raw:
            continue

        if street == 'flop' and cards:
            board_so_far.extend(cards.split())
        elif cards:
            board_so_far.append(cards)
        current_board = " ".join(board_so_far)

        pred_action, conf, _ = get_postflop_prediction(parsed, street)
        if not pred_action:
            continue

        emoji = {'flop': '♣️', 'turn': '♦️', 'river': '♥️'}[street]
        user_action = normalize_action(action_raw)
        pred_norm = normalize_action(pred_action)

        text += f"{emoji} *{street.upper()}* (ваше действие: {user_action.upper()})\n"
        text += f"   🎯 *Рекомендация модели:* {pred_norm.upper()} (уверенность {conf:.0%})\n"

        if current_board:
            texture = analyze_board_texture(current_board)
            text += f"   📋 *Доска:* {current_board} — {texture}\n"
            draws_analysis = calculate_draws_and_odds(last['cards'], current_board, street)
            if draws_analysis:
                text += f"   🧩 *Дро и шансы:* {draws_analysis}\n"

        # Шансы банка
        pot_before = parsed.get(f'{street}_pot', 0)
        call_amount = parsed.get(f'{street}_call_amount', 0)
        if call_amount > 0 and pot_before > 0:
            pot_odds_str = calculate_pot_odds(pot_before, call_amount)
            if pot_odds_str:
                text += f"   {pot_odds_str}\n"
            verdict = pot_odds_vs_equity(pot_before, call_amount, last['cards'], current_board, street)
            if verdict:
                text += f"   {verdict}\n"

        if pred_norm == 'fold':
            text += "   ✨ *Почему:* Рука слабая, опасная доска или много оппонентов.\n"
        elif pred_norm == 'call':
            text += "   ✨ *Почему:* Рука имеет потенциал, но недостаточно сильна для рейза.\n"
        elif pred_norm in ('bet', 'raise'):
            text += "   ✨ *Почему:* Сильная рука или хорошая возможность для блефа/полублефа.\n"
        elif pred_norm == 'check':
            text += "   ✨ *Почему:* Нет необходимости ставить — можно взять бесплатную карту.\n"

        if pred_norm == user_action:
            text += f"   ✅ *Вердикт:* Ваше действие совпало с рекомендацией. Отлично!\n"
        else:
            text += f"   ⚠️ *Вердикт:* Модель советовала {pred_norm.upper()}. Рассмотрите это как альтернативу.\n"
        text += "\n"

    await update.message.reply_text(text, parse_mode='Markdown')

async def postflop_command(update: Update, context: ContextTypes.DEFAULT_TYPE, street: str):
    parsed = context.user_data.get('last_parsed_hand')
    last = context.user_data.get('last_hand_info')
    if not parsed:
        await update.message.reply_text("❌ Нет данных о последней руке. Сначала отправьте раздачу.")
        return

    # Строим полную доску для этой улицы
    board_so_far = []
    for s in ['flop', 'turn', 'river']:
        sc = parsed.get(f'{s}_cards')
        if sc:
            if s == 'flop':
                board_so_far.extend(sc.split())
            else:
                board_so_far.append(sc)
        if s == street:
            break
    current_board = " ".join(board_so_far)
    cards_text = f"🃏 Карты на {street.upper()}: {current_board}" if current_board else f"⚠️ Карты {street.upper()} не найдены."

    action = parsed.get(f'{street}_action')
    if action:
        await update.message.reply_text(
            f"{cards_text}\nℹ️ В этой раздаче на {street.upper()} вы уже совершили действие: {normalize_action(action).upper()}."
        )
        return

    pred_action, confidence, _ = get_postflop_prediction(parsed, street)
    if pred_action is None:
        await update.message.reply_text(f"{cards_text}\n❌ Модель для {street.upper()} не загружена или недостаточно данных.")
        return

    emoji_map = {'fold':'🤚', 'call':'📞', 'bet':'💰', 'raise':'📈', 'check':'✅'}
    pred_norm = normalize_action(pred_action)
    reply = f"{cards_text}\n🎯 *Предсказание на {street.upper()}:* {pred_norm.upper()} {emoji_map.get(pred_norm, '')} (уверенность {confidence:.1%})"

    # Текстура доски
    if current_board:
        texture = analyze_board_texture(current_board)
        if texture:
            reply += f"\n📋 *Текстура:* {texture}"

    # Дро
    if current_board and last:
        draws = calculate_draws_and_odds(last['cards'], current_board, street)
        if draws:
            reply += f"\n\n🧩 *Анализ дро:*\n{draws}"

    # Pot odds
    pot_before = parsed.get(f'{street}_pot', 0)
    call_amount = parsed.get(f'{street}_call_amount', 0)
    if call_amount > 0 and pot_before > 0:
        pot_odds_str = calculate_pot_odds(pot_before, call_amount)
        if pot_odds_str:
            reply += f"\n\n{pot_odds_str}"
        if last:
            verdict = pot_odds_vs_equity(pot_before, call_amount, last['cards'], current_board, street)
            if verdict:
                reply += f"\n{verdict}"

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
    logger.info("Бот запущен v4.0")
    app.run_polling()

if __name__ == "__main__":
    main()
