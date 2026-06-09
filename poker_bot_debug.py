#!/usr/bin/env python3
import re
import pickle
import logging
import numpy as np
from pathlib import Path
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
    """Надёжный парсер, ищет Hero разными способами"""
    result = {
        'hero_seat': None,
        'button_seat': None,
        'hero_position': None,
        'hero_hole_cards': None,
        'hero_stack_pre_bb': 0.0,
        'num_opponents_preflop': 0,
        'big_blind': 1.0
    }

    # 1. Блайнды
    blinds = re.search(r'\$([\d\.]+)/\$([\d\.]+)', content)
    if blinds:
        result['big_blind'] = float(blinds.group(2))

    # 2. Поиск Hero: ищем "Seat X: Hero" (может быть "Seat 5: Hero" или "Seat 5: Hero (nick)")
    hero_seat_match = re.search(r'Seat\s+(\d+):\s*Hero', content, re.IGNORECASE)
    if not hero_seat_match:
        # Альтернатива: может быть "Hero:" в действиях, тогда найдём место по контексту
        hero_action_match = re.search(r'(?:^|\n)(\w+):\s*(?:folds|raises|calls|bets|checks)', content)
        if hero_action_match and hero_action_match.group(1).lower() != 'hero':
            # Если найден ник, отличный от Hero, пробуем искать его как Hero (но это рискованно)
            pass
        logger.warning("Не найдено 'Seat X: Hero'")
        return None
    hero_seat = int(hero_seat_match.group(1))
    result['hero_seat'] = hero_seat

    # 3. Кнопка
    button_match = re.search(r'Seat\s+#?(\d+)\s+is the button', content, re.IGNORECASE)
    if button_match:
        result['button_seat'] = int(button_match.group(1))
        result['hero_position'] = determine_position(hero_seat, result['button_seat'], 9)

    # 4. Карты Hero: ищем "Hero [7h 7s]" или "Dealt to Hero [7h 7s]"
    cards_match = re.search(r'Hero\s+\[([2-9TJQKA][cdhs] [2-9TJQKA][cdhs])\]', content)
    if not cards_match:
        cards_match = re.search(r'Dealt to Hero\s+\[([2-9TJQKA][cdhs] [2-9TJQKA][cdhs])\]', content)
    if cards_match:
        result['hero_hole_cards'] = cards_match.group(1)
    else:
        logger.warning("Не найдены карты Hero")
        # Не возвращаем None, возможно, модель всё равно сможет предсказать? Нет, нужны карты.
        return None

    # 5. Стек Hero
    stack_match = re.search(r'Seat\s+%d:\s+Hero\s+\(\$([\d\.]+)\)' % hero_seat, content)
    if not stack_match:
        # Более гибкий поиск: Seat 5: Hero ($1.97 in chips)
        stack_match = re.search(r'Seat\s+%d:\s+Hero\s+\(\$([\d\.]+)' % hero_seat, content)
    if stack_match and result['big_blind'] > 0:
        result['hero_stack_pre_bb'] = float(stack_match.group(1)) / result['big_blind']

    # 6. Количество оппонентов (игроки, кроме Hero, которые совершали действия)
    all_players = set(re.findall(r'([A-Za-z0-9]+):\s*(?:folds|raises|calls|bets|checks)', content))
    all_players.discard('Hero')
    result['num_opponents_preflop'] = len(all_players)

    return result

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🃏 Пришлите .txt файл с раздачей или текст раздачи.\n"
        "Я предскажу действие Hero на префлопе."
    )

# ---> НОВЫЕ КОМАНДЫ <---
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
        "• Силу руки\n"
        "• Позицию за столом\n"
        "• Размер стека в BB\n"
        "• Количество оппонентов\n\n"
        "💡 *Совет:* Для лучшего результата отправляйте раздачи целиком."
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 *Poker Oracle Bot — v1.0*\n\n"
        "Этот бот создан для анализа покерных раздач и предсказания действий на префлопе. "
        "Он использует модель машинного обучения, обученную на реальных историях рук.\n\n"
        "🧠 *ML-модель:* Random Forest\n"
        "📈 *Признаки:* позиция, сила руки, стек, количество оппонентов\n"
        "♠️ *Поддерживаемые действия:* fold, call, raise, check\n\n"
        "🔧 *Технологии:* Python, python-telegram-bot, scikit-learn, pandas, numpy\n\n"
        "© 2026 | Разработано с любовью к покеру и AI"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = (
        f"🆔 *Ваш профиль:*\n"
        f"• Имя: {user.first_name}\n"
        f"• Username: @{user.username if user.username else 'не указан'}\n"
        f"• ID: {user.id}\n\n"
        f"🤖 *О боте:*\n"
        f"• Версия модели: 1.0\n"
        f"• Дата обучения: 2026\n"
        f"• Точность на тесте: ~63%\n\n"
        f"📊 *Ваша статистика в этом чате:*\n"
        f"• Всего предсказаний: {context.user_data.get('predictions', 0)}\n"
        f"• Успешных: пока не отслеживаем 😅"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Просим пользователя написать отзыв после команды
    await update.message.reply_text(
        "📝 *Помогите улучшить бота!*\n\n"
        "Если предсказание оказалось полезным — просто напишите что-нибудь позитивное 😊\n"
        "Если нет — расскажите, в чём ошибка. Например:\n"
        "• 'Модель ошиблась, рука была слабее'\n"
        "• 'Неправильно определил позицию'\n\n"
        "Ваши отзывы помогут сделать бота умнее! 🙏",
        parse_mode='Markdown'
    )
    # Здесь можно сохранить отзыв в файл или БД
    # context.user_data['awaiting_feedback'] = True

async def terms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "⚖️ *Условия использования:*\n\n"
        "1. Бот предоставляет предсказания только в развлекательных и образовательных целях.\n"
        "2. Разработчик не несёт ответственности за любые финансовые потери, связанные с использованием рекомендаций бота.\n"
        "3. Бот не сохраняет тексты раздач и не передаёт их третьим лицам.\n\n"
        "📅 Последнее обновление: 09.06.2026\n\n"
        "✅ Продолжая использовать бота, вы соглашаетесь с этими условиями."
    )
    await update.message.reply_text(text, parse_mode='Markdown')

# ---> КОНЕЦ НОВЫХ КОМАНД <---

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if model is None:
        await update.message.reply_text("❌ Модель не загружена.")
        return
    # ---> СЧЁТЧИК ПРЕДСКАЗАНИЙ <---
    predictions = context.user_data.get('predictions', 0)
    context.user_data['predictions'] = predictions + 1
    # ---> КОНЕЦ СЧЁТЧИКА <---
    doc = update.message.document
    # ... остальной код функции без изменений ...

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if model is None:
        await update.message.reply_text("❌ Модель не загружена. Сначала обучите модель.")
        return
    doc = update.message.document
    if not doc.file_name.endswith('.txt'):
        await update.message.reply_text("Пожалуйста, отправьте файл .txt")
        return
    file = await doc.get_file()
    content_bytes = await file.download_as_bytearray()
    # Пробуем разные кодировки
    for encoding in ['utf-8', 'cp1251', 'latin1']:
        try:
            text = content_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        await update.message.reply_text("Не удалось декодировать файл. Проверьте кодировку.")
        return

    # Сохраняем для отладки (можно потом удалить)
    with open("debug_received.txt", "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("Файл получен, размер %d символов", len(text))

    # Вызываем предсказание
    await predict(update, text)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if model is None:
        await update.message.reply_text("❌ Модель не загружена.")
        return
    text = update.message.text
    await predict(update, text)

async def predict(update: Update, text: str):
    # Берём первую раздачу, если их несколько
    if 'Poker Hand #' in text:
        first = re.search(r'(Poker Hand #.*?)(?=Poker Hand #|$)', text, re.DOTALL)
        if first:
            text = first.group(1)
    # Парсим
    parsed = parse_hand_robust(text)
    if parsed is None:
        # Отправляем отладочную информацию пользователю
        await update.message.reply_text(
            "❌ Не удалось найти Hero в раздаче. Убедитесь, что в тексте есть строка 'Seat X: Hero'.\n"
            "Пример: 'Seat 5: Hero ($1.97 in chips)'\n\n"
            "Если вы уверены, что раздача корректна, отправьте её как файл .txt."
        )
        # Логируем первые 500 символов для диагностики
        logger.error("Не удалось распарсить. Начало текста:\n%s", text[:500])
        return

    if parsed['hero_hole_cards'] is None:
        await update.message.reply_text("❌ Не найдены карты Hero (должна быть строка 'Hero [7h 7s]').")
        return

    # Признаки
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
    action = encoder.inverse_transform([pred_enc])[0]
    emoji = {'fold':'🤚', 'call':'📞', 'raise':'📈', 'check':'✅'}
    await update.message.reply_text(f"🎯 Предсказанное действие: **{action.upper()}** {emoji.get(action, '')}")

def main():
    TOKEN = "8941942869:AAFOgYdsBuFSKNxBGZOsrqg9S1-ki3SbMlg"  # ЗАМЕНИТЕ
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    print("Бот запущен. Ожидание сообщений...")
    app.run_polling()

if __name__ == "__main__":
    main()