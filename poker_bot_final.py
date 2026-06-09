#!/usr/bin/env python3
import re
import pickle
import logging
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка модели
try:
    with open("poker_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("poker_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    logger.info("✅ Модель и энкодер загружены")
except Exception as e:
    logger.error(f"❌ Ошибка загрузки модели: {e}")
    model = None
    encoder = None

# Константы
RANK_ORDER = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
POS_ORDER = {'BTN':0,'SB':1,'BB':2,'CO':3,'MP3':4,'MP2':5,'MP1':6,'EP':7}
POS_NAMES = {v:k for k,v in POS_ORDER.items()}

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
    if not cards_str:
        return {'is_pair':0,'suited':0,'high_card_rank':0,'low_card_rank':0,'gap':-1,'hand_group':0}
    parts = cards_str.strip().split()
    if len(parts) != 2:
        return {'is_pair':0,'suited':0,'high_card_rank':0,'low_card_rank':0,'gap':-1,'hand_group':0}
    c1, c2 = parts[0], parts[1]
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
    return {'is_pair':is_pair, 'suited':suited, 'high_card_rank':high, 'low_card_rank':low, 'gap':gap, 'hand_group':group}

def parse_hand_robust(content):
    result = {
        'hero_seat': None,
        'button_seat': None,
        'hero_position': None,
        'hero_hole_cards': None,
        'hero_stack_pre_bb': 0.0,
        'num_opponents_preflop': 0,
        'big_blind': 1.0
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
        result['button_seat'] = int(button_match.group(1))
        result['hero_position'] = determine_position(hero_seat, result['button_seat'], 9)
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
    all_players = set(re.findall(r'([A-Za-z0-9]+):\s*(?:folds|raises|calls|bets|checks)', content))
    all_players.discard('Hero')
    result['num_opponents_preflop'] = len(all_players)
    return result

# --- Команды бота ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🃏 Пришлите .txt файл с раздачей или текст раздачи.\n"
        "Я предскажу действие Hero на префлопе.\n\n"
        "Команды:\n/stats — как пользоваться\n/about — о боте\n/profile — ваш профиль\n/analysis — детальный разбор последней руки\n/feedback — отзыв\n/terms — условия"
    )

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🎯 *Как пользоваться ботом:*\n"
        "1. Отправьте мне текстовый файл (`.txt`) с историей раздачи\n"
        "2. Или просто скопируйте и вставьте текст раздачи в чат\n\n"
        "✨ *Что я умею:*\n"
        "• Определяю позицию Hero (BTN, SB, BB, CO и т.д.)\n"
        "• Анализирую стартовую руку (карманная пара, одномастные, коннекторы)\n"
        "• Предсказываю оптимальное действие на префлопе: 🤚 fold, 📞 call, 📈 raise, ✅ check\n\n"
        "📊 *Модель обучена на реальных раздачах и учитывает:*\n"
        "• Силу руки\n• Позицию за столом\n• Размер стека в BB\n• Количество оппонентов"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 *Poker Oracle Bot — v1.0*\n\n"
        "Этот бот использует модель машинного обучения (Random Forest) для предсказания действий на префлопе.\n"
        "Обучен на 190 реальных раздачах (датасет будет расширяться).\n\n"
        "🔧 *Технологии:* Python, python-telegram-bot, scikit-learn, pandas, numpy\n"
        "© 2026 | Разработано с любовью к покеру и AI"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    predictions = context.user_data.get('predictions', 0)
    last_hand_info = context.user_data.get('last_hand_info', None)
    text = (
        f"🆔 *Ваш профиль:*\n"
        f"• Имя: {user.first_name}\n"
        f"• Username: @{user.username if user.username else 'не указан'}\n"
        f"• ID: {user.id}\n\n"
        f"📊 *Статистика:*\n"
        f"• Предсказаний сделано: {predictions}\n"
        f"• Последняя рука: {last_hand_info['action'] if last_hand_info else 'нет'}"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📝 *Помогите улучшить бота!*\n\n"
        "Если предсказание оказалось полезным — просто напишите что-нибудь позитивное 😊\n"
        "Если нет — расскажите, в чём ошибка. Например:\n"
        "• 'Модель ошиблась, рука была слабее'\n"
        "• 'Неправильно определил позицию'\n\n"
        "Ваши отзывы помогут сделать бота умнее! 🙏",
        parse_mode='Markdown'
    )

async def terms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "⚖️ *Условия использования:*\n\n"
        "1. Бот предоставляет предсказания только в развлекательных и образовательных целях.\n"
        "2. Разработчик не несёт ответственности за любые финансовые потери, связанные с использованием рекомендаций бота.\n"
        "3. Бот не сохраняет тексты раздач и не передаёт их третьим лицам.\n\n"
        "✅ Продолжая использовать бота, вы соглашаетесь с этими условиями."
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Детальный разбор последней руки"""
    last = context.user_data.get('last_hand_info', None)
    if not last:
        await update.message.reply_text("❌ Нет данных о последней руке. Сначала отправьте раздачу.")
        return
    text = (
        f"🔍 *Детальный разбор последней руки*\n\n"
        f"🃏 *Карты:* {last['cards']}\n"
        f"📌 *Позиция:* {last['position']}\n"
        f"💰 *Стек в BB:* {last['stack_bb']:.1f}\n"
        f"👥 *Оппонентов на префлопе:* {last['opponents']}\n\n"
        f"📊 *Признаки модели:*\n"
        f"• Пара: {'да' if last['is_pair'] else 'нет'}\n"
        f"• Одномастные: {'да' if last['suited'] else 'нет'}\n"
        f"• Старшая карта: {last['high_rank']}\n"
        f"• Младшая карта: {last['low_rank']}\n"
        f"• Разрыв (gap): {last['gap']}\n"
        f"• Группа руки (1-10): {last['hand_group']}\n\n"
        f"🤖 *Предсказание модели:* **{last['action'].upper()}**\n"
        f"📈 *Уверенность:* {last['confidence']:.1%}"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

# --- Обработка раздач и предсказание ---
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if model is None:
        await update.message.reply_text("❌ Модель не загружена.")
        return
    predictions = context.user_data.get('predictions', 0)
    context.user_data['predictions'] = predictions + 1
    doc = update.message.document
    if not doc.file_name.endswith('.txt'):
        await update.message.reply_text("Пожалуйста, отправьте файл .txt")
        return
    file = await doc.get_file()
    content_bytes = await file.download_as_bytearray()
    for encoding in ['utf-8', 'cp1251', 'latin1']:
        try:
            text = content_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        await update.message.reply_text("Не удалось декодировать файл.")
        return
    await predict(update, text, context)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if model is None:
        await update.message.reply_text("❌ Модель не загружена.")
        return
    text = update.message.text
    await predict(update, text, context)

async def predict(update: Update, text: str, context: ContextTypes.DEFAULT_TYPE):
    if 'Poker Hand #' in text:
        first = re.search(r'(Poker Hand #.*?)(?=Poker Hand #|$)', text, re.DOTALL)
        if first:
            text = first.group(1)
    parsed = parse_hand_robust(text)
    if parsed is None:
        await update.message.reply_text("❌ Не удалось найти Hero в раздаче.")
        return
    if parsed['hero_hole_cards'] is None:
        await update.message.reply_text("❌ Не найдены карты Hero.")
        return
    card_feats = hand_features(parsed['hero_hole_cards'])
    pos_num = POS_ORDER.get(parsed['hero_position'], 7) if parsed['hero_position'] else 7
    features = [
        pos_num,
        parsed['hero_stack_pre_bb'],
        parsed['num_opponents_preflop'],
        card_feats['is_pair'],
        card_feats['suited'],
        card_feats['high_card_rank'],
        card_feats['low_card_rank'],
        card_feats['gap'],
        card_feats['hand_group']
    ]
    X = np.array([features])
    pred_enc = model.predict(X)[0]
    # Уверенность: максимальная вероятность среди классов
    probs = model.predict_proba(X)[0]
    confidence = np.max(probs)
    action = encoder.inverse_transform([pred_enc])[0]
    emoji = {'fold':'🤚', 'call':'📞', 'raise':'📈', 'check':'✅'}
    await update.message.reply_text(f"🎯 Предсказанное действие: **{action.upper()}** {emoji.get(action, '')}\n(уверенность: {confidence:.1%})")
    
    # Сохраняем информацию для /analysis
    context.user_data['last_hand_info'] = {
        'cards': parsed['hero_hole_cards'],
        'position': parsed['hero_position'],
        'stack_bb': parsed['hero_stack_pre_bb'],
        'opponents': parsed['num_opponents_preflop'],
        'is_pair': card_feats['is_pair'],
        'suited': card_feats['suited'],
        'high_rank': card_feats['high_card_rank'],
        'low_rank': card_feats['low_card_rank'],
        'gap': card_feats['gap'],
        'hand_group': card_feats['hand_group'],
        'action': action,
        'confidence': confidence
    }

def main():
    TOKEN = "8941942869:AAFOgYdsBuFSKNxBGZOsrqg9S1-ki3SbMlg"  # ЗАМЕНИТЕ
    app = Application.builder().token(TOKEN).build()
    # Регистрируем все команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(CommandHandler("profile", profile))
    app.add_handler(CommandHandler("feedback", feedback))
    app.add_handler(CommandHandler("terms", terms))
    app.add_handler(CommandHandler("analysis", analysis))
    # Обработка сообщений (текст и файлы)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    print("Бот запущен. Все команды активны.")
    app.run_polling()

if __name__ == "__main__":
    main()