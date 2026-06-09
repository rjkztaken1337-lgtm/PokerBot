# app.py
import os
import threading
from flask import Flask
from poker_bot_universal import main

app = Flask(__name__)

@app.route('/')
def health():
    return "Bot is running"

@app.route('/health')
def health_check():
    return "OK", 200

def run_bot():
    main()

if __name__ == '__main__':
    # Запускаем бота в отдельном фоновом потоке
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()

    # Запускаем Flask-сервер на порту, который назначает Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)