#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poker Oracle Bot - Final Version
Commands: /start, /stats, /about, /profile, /analysis, /explain, /feedback, /terms, /reload
"""

import re
import pickle
import logging
import sqlite3
import numpy as np
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes

# ========== НАСТРОЙКА ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== КОНФИГУРАЦИЯ ==========
TOKEN = "YOUR_BOT_TOKEN_HERE"           # ЗАМЕНИТЕ НА РЕАЛЬНЫЙ ТОКЕН
ADMIN_ID = 123456789                    # ЗАМЕНИТЕ НА ВАШ TELEGRAM ID (узнайте у @userinfobot)
DB_PATH = "/Users/user/Desktop/PokerBot/feedback.db"   # ПУТЬ К БАЗЕ ДАННЫХ

# ========== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ МОДЕЛИ ==========
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

# ========== БАЗА ДАННЫХ ==========
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

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
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
    # Группа руки (приближённо)
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
    # Блайнды
    blinds = re.search(r'\$([\d\.]+)/\$([\d\.]+)', content)
    if blinds:
        result['big_blind'] = float(blinds.group(2))
    # Поиск Hero
    hero_seat_match = re.search(r'Seat\s+(\d+):\s*Hero', content, re.IGNORECASE)
    if not hero_seat_match:
        return None
    hero_seat = int(hero_seat_match.group(1))
    result['hero_seat'] = hero_seat
    # Кнопка
    button_match = re.search(r'Seat\s+#?(\d+)\s+is the button', content, re.IGNORECASE)
    if button_match:
        button_seat = int(button_match.group(1))
        result['hero_position'] = determine_position(hero_seat, button_seat, 9)
    # Карты
    cards_match = re.search(r'Hero\s+\[([2-9TJQKA][cdhs] [2-9TJQKA][cdhs])\]', content)
    if not cards_match:
        cards_match = re.search(r'Dealt to Hero\s+\[([2-9TJQKA][cdhs] [2-9TJQKA][cdhs])\]', content)
    if cards_match:
        result['hero_hole_cards'] = cards_match.group(1)
    else:
        return None
    # Стек
    stack_match = re.search(r'Seat\s+%d:\s+Hero\s+\(\$([\d\.]+)' % hero_seat, content)
    if stack_match and result['big_blind'] > 0:
        result['hero_stack_pre_bb'] = float(stack_match.group(1)) / result['big_blind']
    # Оппоненты
    players = set(re.findall(r'([A-Za-z0-9]+):\s*(?:folds|raises|calls|bets|checks)', content))
    players.discard('Hero')
    result['num_opponents_preflop'] = len(players)
    return result

# ========== КОМАНДЫ ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🃏 *Poker Oracle Bot*\n\n"
        "Отправьте мне текстовый файл (.txt) с раздачей или просто вставьте текст.\n"
        "Я определю позицию Hero, проанализирую руку и предскажу оптимальное действие.\n\n"
        "📌 *Команды:*\n"
        "/stats – как пользоваться\n"
        "/about – о боте\n"
        "/profile – ваша статистика\n"
        "/analysis – детали последней руки\n"
        "/explain – почему модель решила именно так\n"
        "/feedback – оставить отзыв\n"
        "/terms – условия использования",
        parse_mode='Markdown'
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
        "/analysis – подробный разбор последней руки\n"
        "/explain – понятное объяснение, почему бот дал такой совет"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 *Poker Oracle Bot v2.0*\n\n"
        "🧠 *Модель:* Random Forest, обучена на реальных раздачах.\n"
        "📊 *Признаки:* позиция, сила руки, стек в BB, количество оппонентов.\n"
        "♠️ *Поддерживаемые румы:* PokerStars, GG Poker, PartyPoker.\n"
        "📈 *Функции:* предсказание действий, сбор обратной связи, дообучение, объяснение решений.\n\n"
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
        f"👥 *Оппонентов:* {last['opponents']}\n\n"
        f"📊 *Признаки:*\n"
        f"• Пара: {'да' if last['is_pair'] else 'нет'}\n"
        f"• Одномастные: {'да' if last['suited'] else 'нет'}\n"
        f"• Старшая карта: {last['high_card']}\n"
        f"• Младшая карта: {last['low_card']}\n"
        f"• Разрыв (gap): {last['gap']}\n"
        f"• Группа руки: {last['hand_group']}\n\n"
        f"🤖 *Предсказание:* **{last['action'].upper()}** (уверенность {last['confidence']:.1%})"
    )
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
    high = info['high_card']       # уже буква
    low = info['low_card']

    reasons = []
    # 1. Сила руки
    if is_pair:
        if high in ['A','K','Q','J','T'] and high != 'T':
            reasons.append(f"🔹 У вас сильная карманная пара ({high}{high}) – это хорошая рука.")
        elif high in ['9','8','7']:
            reasons.append(f"🔹 Средняя пара ({high}{high}) – играбельно, но осторожно.")
        else:
            reasons.append(f"🔹 Слабая пара ({high}{high}) – часто проигрывает старшим картам.")
    else:
        if suited and abs(info['high_rank'] - info['low_rank']) <= 2 and info['high_rank'] >= 12:
            reasons.append(f"🔹 Отличная одномастная рука ({high}{low} одномастные) – можно разыгрывать агрессивно.")
        elif suited and info['high_rank'] >= 10:
            reasons.append(f"🔹 Хорошая одномастная рука ({high}{low} одномастные).")
        elif info['high_rank'] >= 12:
            reasons.append(f"🔹 Старшие карты ({high}{low}) – неплохо, но не одномастные.")
        else:
            reasons.append(f"🔹 Слабая рука ({high}{low}, не пара, не одномастные) – лучше сбросить.")
    # 2. Позиция
    if position in ['EP', 'MP1', 'MP2']:
        reasons.append(f"🔹 Вы на ранней позиции ({position}) – нужно играть только сильные руки.")
    elif position in ['MP3', 'CO']:
        reasons.append(f"🔹 Средняя позиция ({position}) – диапазон можно расширить.")
    elif position == 'BB':
        reasons.append(f"🔹 Вы на большом блайнде – уже вложили деньги, можно защищаться с подходящей рукой.")
    else:  # BTN, SB
        reasons.append(f"🔹 Вы на поздней позиции ({position}) – преимущество, можно играть больше рук.")
    # 3. Количество оппонентов
    if opponents >= 4:
        reasons.append(f"🔹 За столом {opponents} игроков – много соперников, нужна действительно сильная рука.")
    elif opponents >= 2:
        reasons.append(f"🔹 {opponents} оппонента – средняя опасность.")
    else:
        reasons.append(f"🔹 Всего {opponents} оппонент – можно играть шире.")
    # 4. Стек
    if stack < 20:
        reasons.append(f"🔹 Ваш стек всего {stack:.0f} BB – короткая стопка, либо оллин, либо фолд.")
    # 5. Итоговая рекомендация
    conclusion = {
        'FOLD': "🎯 **Модель советует СБРОСИТЬ**. Слишком много факторов против розыгрыша.",
        'CALL': "🎯 **Модель советует УРАВНЯТЬ**. Рука имеет потенциал, но не настолько сильная для рейза.",
        'RAISE': "🎯 **Модель советует ПОВЫСИТЬ**. Рука и позиция позволяют атаковать.",
        'CHECK': "🎯 **Модель советует ЧЕКНУТЬ**. В данной ситуации нет смысла ставить."
    }.get(action, "")
    text = f"🔍 *Объяснение предсказания: {action}* (уверенность {confidence:.0%})\n\n"
    text += "\n".join(reasons)
    text += f"\n\n{conclusion}\n\n💡 *В итоге:* {cards}, {position}, {opponents} оппонентов, стек {stack:.0f} BB – модель выбрала {action}."
    await update.message.reply_text(text, parse_mode='Markdown')

async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📝 Напишите ваш отзыв или пожелание. Администратор получит его.")

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
    if update.effective_user.id != 1246001390:
        await update.message.reply_text("⛔ Доступ запрещён.")
        return
    if load_model_files():
        await update.message.reply_text("✅ Модель успешно перезагружена из файлов.")
    else:
        await update.message.reply_text("❌ Ошибка перезагрузки модели.")

# ========== ПРЕДСКАЗАНИЯ ==========
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if model is None:
        await update.message.reply_text("❌ Модель не загружена. Попробуйте позже.")
        return
    doc = update.message.document
    if not doc.file_name.endswith('.txt'):
        await update.message.reply_text("Пожалуйста, отправьте файл в формате .txt")
        return
    file = await doc.get_file()
    content_bytes = await file.download_as_bytearray()
    for enc in ['utf-8', 'cp1251', 'latin1']:
        try:
            text = content_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        await update.message.reply_text("Не удалось прочитать файл. Попробуйте другой.")
        return
    await predict(update, text, context)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if model is None:
        await update.message.reply_text("❌ Модель не загружена.")
        return
    text = update.message.text
    if text.startswith('/'):
        return
    await predict(update, text, context)

async def predict(update: Update, text: str, context: ContextTypes.DEFAULT_TYPE):
    # Выделяем первую раздачу
    if 'Poker Hand #' in text:
        first = re.search(r'(Poker Hand #.*?)(?=Poker Hand #|$)', text, re.DOTALL)
        if first:
            text = first.group(1)
    parsed = parse_hand_robust(text)
    if not parsed or not parsed['hero_hole_cards']:
        await update.message.reply_text("❌ Не удалось распознать раздачу. Убедитесь, что формат подходит (PokerStars, GG Poker).")
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
    emoji = {'fold':'🤚', 'call':'📞', 'raise':'📈', 'check':'✅'}
    reply = f"🎯 *Предсказанное действие:* {action.upper()} {emoji.get(action, '')}\n(уверенность {confidence:.1%})"
    
    # Сохраняем в БД
    pred_id = save_prediction(update.effective_user.id, text[:1000], features, action, feedback=None)
    keyboard = [[InlineKeyboardButton("✅ Верно", callback_data=f"feedback_{pred_id}_yes"),
                 InlineKeyboardButton("❌ Неверно", callback_data=f"feedback_{pred_id}_no")]]
    
    await update.message.reply_text(reply, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
    
    # Сохраняем в context.user_data
    # Преобразуем ранги обратно в буквы для отображения
    high_card = rank_to_letter.get(card_feats['high_card_rank'], str(card_feats['high_card_rank']))
    low_card = rank_to_letter.get(card_feats['low_card_rank'], str(card_feats['low_card_rank']))
    context.user_data['last_hand_info'] = {
        'cards': parsed['hero_hole_cards'],
        'position': parsed['hero_position'],
        'stack_bb': parsed['hero_stack_pre_bb'],
        'opponents': parsed['num_opponents_preflop'],
        'is_pair': card_feats['is_pair'],
        'suited': card_feats['suited'],
        'high_card': high_card,
        'low_card': low_card,
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
            await query.message.reply_text("🙏 Спасибо за обратную связь!")

# ========== ЗАПУСК ==========
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

if __name__ == "__main__":
    main()