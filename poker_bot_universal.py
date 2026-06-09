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

# ---------- Токен и админ из переменных окружения ----------
TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_TOKEN not set")
ADMIN_ID = int(os.environ.get("ADMIN_ID", 0))

# ---------- Глобальные переменные ----------
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

# ---------- Константы и парсер (расширенный) ----------
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

def parse_hand_advanced(content):
    """
    Расширенный парсер, извлекающий префлоп и постфлоп действия Hero.
    Возвращает словарь с полями:
      - hero_position, hero_hole_cards, hero_stack_pre_bb, big_blind
      - preflop_action, preflop_opponents
      - flop_action, flop_opponents, flop_cbet
      - turn_action, turn_opponents
      - river_action, river_opponents
    """
    result = {
        'hero_seat': None,
        'hero_position': None,
        'hero_hole_cards': None,
        'big_blind': 1.0,
        'hero_stack_pre_bb': 0.0,
        # Префлоп
        'preflop_action': None,
        'preflop_opponents': 0,
        # Флоп
        'flop_action': None,
        'flop_opponents': 0,
        'flop_cbet': 0,
        # Тёрн
        'turn_action': None,
        'turn_opponents': 0,
        # Ривер
        'river_action': None,
        'river_opponents': 0,
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
    
    # ---- Извлечение действий по улицам ----
    sections = {
        'preflop': ('*** HOLE CARDS ***', '*** FLOP ***'),
        'flop': ('*** FLOP ***', '*** TURN ***'),
        'turn': ('*** TURN ***', '*** RIVER ***'),
        'river': ('*** RIVER ***', '*** SHOW DOWN ***')
    }
    for street, (start_marker, end_marker) in sections.items():
        start = content.find(start_marker)
        if start == -1:
            continue
        end = content.find(end_marker, start) if end_marker else len(content)
        block = content[start:end]
        # Действие Hero
        pattern = r'Hero:\s*(folds|checks|calls|bets|raises)(?:\s*(?:\$?[\d\.]+)?\s*(?:to\s*\$?[\d\.]+)?)?'
        match = re.search(pattern, block, re.IGNORECASE)
        if match:
            action = match.group(1)
            result[f'{street}_action'] = action
        # Количество оппонентов
        players = set(re.findall(r'([A-Za-z0-9_]+):\s*(?:folds|raises|calls|bets|checks)', block))
        players.discard('Hero')
        result[f'{street}_opponents'] = len(players)
    
    # Конт-бет на флопе: если Hero сделал bet или raise на флопе
    if result['flop_action'] in ['bets', 'raises']:
        result['flop_cbet'] = 1
    return result

# ---------- Команды ----------
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
        "/feedback — отзыв\n"
        "/terms — условия\n"
        "/reload — перезагрузить модель (только админ)\n"
        "/retrain_now — переобучить модель (только админ)"
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
        "/analysis – подробный разбор последней руки (включая постфлоп)\n"
        "/explain – понятное объяснение, почему бот дал такой совет"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 *Poker Oracle Bot v2.1* (с поддержкой постфлопа)\n\n"
        "🧠 *Модель:* Random Forest, обучена на реальных раздачах.\n"
        "📊 *Признаки:* позиция, сила руки, стек в BB, количество оппонентов.\n"
        "♠️ *Поддерживаемые румы:* PokerStars, GG Poker, PartyPoker.\n"
        "📈 *Функции:* предсказание действий, сбор обратной связи, дообучение, объяснение решений.\n"
        "🃏 *Новое:* анализ действий на флопе, тёрне и ривере в команде /analysis.\n\n"
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
    if not last:
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
    # Добавляем постфлоп информацию, если есть
    if 'flop_action' in last and last['flop_action']:
        text += f"\n♣️ *Флоп:* действие Hero — {last['flop_action'].upper()}, оппонентов — {last['flop_opponents']}"
        if last.get('flop_cbet'):
            text += " (конт-бет ✅)"
    if 'turn_action' in last and last['turn_action']:
        text += f"\n♦️ *Тёрн:* действие Hero — {last['turn_action'].upper()}, оппонентов — {last['turn_opponents']}"
    if 'river_action' in last and last['river_action']:
        text += f"\n♥️ *Ривер:* действие Hero — {last['river_action'].upper()}, оппонентов — {last['river_opponents']}"
    await update.message.reply_text(text, parse_mode='Markdown')

async def explain_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info = context.user_data.get('last_hand_info')
    if not info:
        await update.message.reply_text("❌ Нет данных для объяснения. Сначала отправьте раздачу.")
        return
    action = info['action'].upper()
    confidence = info['confidence']
    cards = info['cards']
    position = info['position']
    opponents = info['opponents']
    stack = info['stack_bb']
    is_pair = info['is_pair']
    suited = info['suited']
    high = info['high_card']
    low = info['low_card']

    reasons = []
    if is_pair:
        if high in ['A','K','Q','J','T']:
            reasons.append(f"🔹 Сильная карманная пара ({high}{high})")
        else:
            reasons.append(f"🔹 Слабая пара ({high}{high})")
    else:
        if suited and high in ['A','K','Q','J']:
            reasons.append(f"🔹 Хорошая одномастная рука ({high}{low} одномастные)")
        else:
            reasons.append(f"🔹 Слабая рука ({high}{low}, не пара, не одномастные)")
    if position in ['EP','MP1','MP2']:
        reasons.append(f"🔹 Ранняя позиция ({position}) – рискованно")
    elif position in ['MP3','CO']:
        reasons.append(f"🔹 Средняя позиция ({position})")
    else:
        reasons.append(f"🔹 Поздняя позиция ({position}) – преимущество")
    if opponents >= 4:
        reasons.append(f"🔹 Много оппонентов ({opponents}) – нужна сильная рука")
    if stack < 20:
        reasons.append(f"🔹 Короткий стек ({stack:.0f} BB) – либо оллин, либо фолд")
    conclusion = {
        'FOLD': '🎯 Модель советует **СБРОСИТЬ**',
        'CALL': '🎯 Модель советует **УРАВНЯТЬ**',
        'RAISE': '🎯 Модель советует **ПОВЫСИТЬ**',
        'CHECK': '🎯 Модель советует **ЧЕКНУТЬ**'
    }.get(action, '')
    text = f"🔍 *Объяснение:* {action} (уверенность {confidence:.0%})\n\n"
    text += "\n".join(reasons)
    text += f"\n\n{conclusion}"
    await update.message.reply_text(text, parse_mode='Markdown')

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
    if load_model_files():
        await update.message.reply_text("✅ Модель перезагружена из файлов.")
        logger.info("Модель перезагружена админом")
    else:
        await update.message.reply_text("❌ Ошибка перезагрузки модели.")

async def retrain_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("⛔ Доступ запрещён.")
        return
    await update.message.reply_text("🔄 Запускаю переобучение модели (может занять до минуты)...")
    try:
        # Запускаем auto_retrain.py
        result = subprocess.run([sys.executable, "auto_retrain.py"], capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            await update.message.reply_text("✅ Переобучение завершено. Перезагружаю модель...")
            if load_model_files():
                await update.message.reply_text("✅ Модель успешно перезагружена.")
            else:
                await update.message.reply_text("❌ Модель переобучена, но не удалось загрузить файлы модели.")
        else:
            await update.message.reply_text(f"❌ Ошибка переобучения:\n{result.stderr[:1000]}")
    except subprocess.TimeoutExpired:
        await update.message.reply_text("❌ Превышено время ожидания (120 секунд).")
    except Exception as e:
        await update.message.reply_text(f"❌ Исключение: {e}")

# ---------- Обработка предсказаний (с расширенным парсером) ----------
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
    
    # Режим ожидания отзыва
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
    
    # Обычная обработка раздачи
    await predict(update, text, context)
    
async def predict(update: Update, text: str, context: ContextTypes.DEFAULT_TYPE):
    # Извлекаем первую раздачу
    if 'Poker Hand #' in text:
        first = re.search(r'(Poker Hand #.*?)(?=Poker Hand #|$)', text, re.DOTALL)
        if first:
            text = first.group(1)
    # Используем расширенный парсер
    parsed = parse_hand_advanced(text)
    if not parsed or not parsed['hero_hole_cards']:
        await update.message.reply_text("❌ Не удалось распознать раздачу. Убедитесь в формате.")
        return
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
    # Сохраняем полную информацию, включая постфлоп
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
        # Постфлоп
        'flop_action': parsed['flop_action'],
        'flop_opponents': parsed['flop_opponents'],
        'flop_cbet': parsed['flop_cbet'],
        'turn_action': parsed['turn_action'],
        'turn_opponents': parsed['turn_opponents'],
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

# ---------- Запуск ----------
def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(CommandHandler("profile", profile))
    app.add_handler(CommandHandler("analysis", analysis))
    app.add_handler(CommandHandler("explain", explain_command))
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
