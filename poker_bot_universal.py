#!/usr/bin/env python3
import os
import re
import pickle
import logging
import sqlite3
import numpy as np
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes

# ---------- Настройка ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- Токен из переменной окружения ----------
TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("Переменная окружения TELEGRAM_TOKEN не установлена")

# ---------- Глобальные переменные модели ----------
model = None
encoder = None

def load_model_files():
    global model, encoder
    try:
        with open("poker_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("poker_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        logger.info("Модель и энкодер загружены")
        return True
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        model = None
        encoder = None
        return False

load_model_files()

# ---------- База данных ----------
DB_PATH = "feedback.db"  # будет создана в текущей папке (Render)

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

# ---------- Константы и парсер ----------
RANK_ORDER = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
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

def parse_hand_robust(content):
    result = {
        'hero_seat': None, 'hero_position': None, 'hero_hole_cards': None,
        'hero_stack_pre_bb': 0.0, 'num_opponents_preflop': 0, 'big_blind': 1.0
    }
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
    players = set(re.findall(r'([A-Za-z0-9]+):\s*(?:folds|raises|calls|bets|checks)', content))
    players.discard('Hero')
    result['num_opponents_preflop'] = len(players)
    return result

# ---------- Команды бота ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🃏 *Poker Oracle Bot*\n\n"
        "Отправьте мне текстовый файл (.txt) с раздачей или просто вставьте текст.\n"
        "Я предскажу действие Hero на префлопе.\n\n"
        "Команды: /stats, /about, /profile, /analysis, /explain, /feedback, /terms",
        parse_mode='Markdown'
    )

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🎯 *Как пользоваться*\n\n"
        "1. Отправьте раздачу (файл .txt или текст).\n"
        "2. Получите предсказание (fold/call/raise/check).\n"
        "3. Оцените точность кнопками ✅/❌.\n"
        "4. Используйте /explain для понятного объяснения."
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "🤖 *Poker Oracle Bot v2.0*\n\nМодель Random Forest, обучена на реальных раздачах. Поддерживает PokerStars, GG Poker, PartyPoker."
    await update.message.reply_text(text, parse_mode='Markdown')

async def profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    preds = context.user_data.get('predictions', 0)
    last = context.user_data.get('last_hand_info', {})
    last_action = last.get('action', 'нет').upper()
    await update.message.reply_text(f"🆔 Ваш ID: {update.effective_user.id}\n📊 Предсказаний: {preds}\n🎯 Последнее действие: {last_action}")

async def analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info = context.user_data.get('last_hand_info')
    if not info:
        await update.message.reply_text("Нет данных. Сначала отправьте раздачу.")
        return
    text = (
        f"🔍 *Детальный разбор*\n\n"
        f"🃏 Карты: {info['cards']}\n"
        f"📌 Позиция: {info['position']}\n"
        f"💰 Стек в BB: {info['stack_bb']:.1f}\n"
        f"👥 Оппонентов: {info['opponents']}\n"
        f"🤖 Предсказание: **{info['action'].upper()}** (уверенность {info['confidence']:.1%})"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def explain_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info = context.user_data.get('last_hand_info')
    if not info:
        await update.message.reply_text("Нет данных для объяснения. Сначала отправьте раздачу.")
        return
    reasons = []
    if info['is_pair']:
        reasons.append(f"🔹 Карманная пара {info['high_card']}{info['high_card']} – {'сильная' if info['high_card'] in 'AKQJT' else 'средняя/слабая'}")
    else:
        suited_str = "одномастные" if info['suited'] else "разномастные"
        reasons.append(f"🔹 {info['high_card']}{info['low_card']} {suited_str} – {'хорошая' if info['suited'] and info['high_rank']>=12 else 'слабая'}")
    pos_desc = {"EP":"ранняя позиция (EP) – рискованно", "MP1/MP2/MP3":"средняя позиция", "CO/BTN/SB/BB":"поздняя позиция – преимущество"}
    reasons.append(f"🔹 Позиция {info['position']} – {pos_desc.get(info['position'], 'средняя')}")
    if info['opponents'] >= 4:
        reasons.append(f"🔹 Много оппонентов ({info['opponents']}) – нужна сильная рука")
    if info['stack_bb'] < 20:
        reasons.append(f"🔹 Короткий стек ({info['stack_bb']:.0f} BB)")
    conclusion = {
        'fold': "🎯 Модель советует **СБРОСИТЬ**",
        'call': "🎯 Модель советует **УРАВНЯТЬ**",
        'raise': "🎯 Модель советует **ПОВЫСИТЬ**",
        'check': "🎯 Модель советует **ЧЕКНУТЬ**"
    }.get(info['action'], "")
    text = f"🔍 *Объяснение:* {info['action'].upper()} (уверенность {info['confidence']:.0%})\n\n" + "\n".join(reasons) + f"\n\n{conclusion}"
    await update.message.reply_text(text, parse_mode='Markdown')

async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Напишите ваш отзыв или предложение. Администратор получит.")

async def terms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "⚖️ Условия: предсказания носят развлекательный характер. Используя бота, вы соглашаетесь."
    await update.message.reply_text(text)

async def reload_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ADMIN_ID = int(os.environ.get("1246001390", 0))  # задайте в Render переменную ADMIN_ID
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("Доступ запрещён.")
        return
    if load_model_files():
        await update.message.reply_text("Модель перезагружена")
    else:
        await update.message.reply_text("Ошибка перезагрузки")

# ---------- Обработчики ----------
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if model is None:
        await update.message.reply_text("Модель не загружена")
        return
    doc = update.message.document
    if not doc.file_name.endswith('.txt'):
        await update.message.reply_text("Требуется .txt файл")
        return
    file = await doc.get_file()
    content_bytes = await file.download_as_bytearray()
    try:
        text = content_bytes.decode('utf-8')
    except:
        await update.message.reply_text("Ошибка декодирования")
        return
    await predict(update, text, context)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if model is None:
        await update.message.reply_text("Модель не загружена")
        return
    text = update.message.text
    if text.startswith('/'):
        return
    await predict(update, text, context)

async def predict(update: Update, text: str, context: ContextTypes.DEFAULT_TYPE):
    if 'Poker Hand #' in text:
        m = re.search(r'(Poker Hand #.*?)(?=Poker Hand #|$)', text, re.DOTALL)
        if m:
            text = m.group(1)
    parsed = parse_hand_robust(text)
    if not parsed or not parsed['hero_hole_cards']:
        await update.message.reply_text("Не удалось распознать раздачу.")
        return
    card_feats = hand_features(parsed['hero_hole_cards'])
    pos_num = POS_ORDER.get(parsed['hero_position'], 7)
    features = [
        pos_num, parsed['hero_stack_pre_bb'], parsed['num_opponents_preflop'],
        card_feats['is_pair'], card_feats['suited'],
        card_feats['high_card_rank'], card_feats['low_card_rank'],
        card_feats['gap'], card_feats['hand_group']
    ]
    X = np.array([features])
    pred_enc = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    confidence = np.max(probs)
    action = encoder.inverse_transform([pred_enc])[0]
    emoji = {'fold':'🤚','call':'📞','raise':'📈','check':'✅'}
    reply = f"🎯 *Предсказание:* {action.upper()} {emoji.get(action,'')} (уверенность {confidence:.1%})"
    
    pred_id = save_prediction(update.effective_user.id, text[:1000], features, action, None)
    keyboard = [[InlineKeyboardButton("✅ Верно", callback_data=f"feedback_{pred_id}_yes"),
                 InlineKeyboardButton("❌ Неверно", callback_data=f"feedback_{pred_id}_no")]]
    await update.message.reply_text(reply, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
    
    high_char = rank_to_letter.get(card_feats['high_card_rank'], str(card_feats['high_card_rank']))
    low_char = rank_to_letter.get(card_feats['low_card_rank'], str(card_feats['low_card_rank']))
    context.user_data['last_hand_info'] = {
        'cards': parsed['hero_hole_cards'],
        'position': parsed['hero_position'],
        'stack_bb': parsed['hero_stack_pre_bb'],
        'opponents': parsed['num_opponents_preflop'],
        'is_pair': card_feats['is_pair'],
        'suited': card_feats['suited'],
        'high_card': high_char,
        'low_card': low_char,
        'high_rank': card_feats['high_card_rank'],
        'low_rank': card_feats['low_card_rank'],
        'gap': card_feats['gap'],
        'hand_group': card_feats['hand_group'],
        'action': action,
        'confidence': confidence
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
            await query.message.reply_text("Спасибо за оценку!")

def main():
    app = Application.builder().token("8941942869:AAFOgYdsBuFSKNxBGZOsrqg9S1-ki3SbMlg").build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(CommandHandler("profile", profile))
    app.add_handler(CommandHandler("analysis", analysis))
    app.add_handler(CommandHandler("explain", explain_command))
    app.add_handler(CommandHandler("feedback", feedback_command))
    app.add_handler(CommandHandler("terms", terms))
    app.add_handler(CommandHandler("reload", reload_model))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(callback_handler))
    logger.info("Бот запущен")
    app.run_polling()

if __name__ == '__main__':
    main()