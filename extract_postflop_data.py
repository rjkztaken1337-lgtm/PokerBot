#!/usr/bin/env python3
import os
import re
import pandas as pd
from pathlib import Path

# ---------- Копия необходимых функций из бота (без зависимостей от телеграма) ----------
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

# ---------- Основной скрипт ----------
def process_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    hands = re.split(r'(?=Poker Hand #)', content)
    records = []
    for hand_text in hands:
        if 'Hero' not in hand_text:
            continue
        parsed = parse_hand_advanced(hand_text)
        if not parsed or not parsed['hero_hole_cards']:
            continue
        card_feats = hand_features(parsed['hero_hole_cards'])
        pos_num = POS_ORDER.get(parsed['hero_position'], 7)
        stack_bb = parsed['hero_stack_pre_bb']
        
        for street in ['flop', 'turn', 'river']:
            action_key = f'{street}_action'
            action = parsed.get(action_key)
            if not action:
                continue
            row = {
                'street': street,
                'hero_position_num': pos_num,
                'hero_stack_pre_bb': stack_bb,
                'preflop_opponents': parsed.get('preflop_opponents', 0),
                'is_pair': card_feats['is_pair'],
                'suited': card_feats['suited'],
                'high_card_rank': card_feats['high_card_rank'],
                'low_card_rank': card_feats['low_card_rank'],
                'gap': card_feats['gap'],
                'hand_group': card_feats['hand_group'],
                'opponents': parsed.get(f'{street}_opponents', 0),
                'action': action,
            }
            prev_streets = {'flop': None, 'turn': 'flop', 'river': 'turn'}
            prev = prev_streets.get(street)
            if prev:
                row[f'{prev}_action'] = parsed.get(f'{prev}_action', 'none')
            records.append(row)
    return records

def main():
    folder = "/Users/user/Desktop/PokerBot"  # замените на нужную папку с TXT файлами
    txt_files = list(Path(folder).glob("*.txt")) + list(Path(folder).glob("*.TXT"))
    if not txt_files:
        print("Нет TXT файлов в указанной папке. Проверьте путь.")
        return
    all_records = []
    for file_path in txt_files:
        print(f"Обработка {file_path.name}...")
        records = process_txt_file(file_path)
        all_records.extend(records)
    df = pd.DataFrame(all_records)
    df.to_csv("postflop_dataset.csv", index=False)
    print(f"Сохранено {len(df)} записей (флоп/тёрн/ривер).")

if __name__ == "__main__":
    main()