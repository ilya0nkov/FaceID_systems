import json
import os
from pathlib import Path

# Используем абсолютный путь относительно расположения этого файла
DB_DIR = os.path.join(os.path.dirname(__file__), '..')
DB_FILE = os.path.join(DB_DIR, 'face_db.json')


def load_db():
    """Загружает базу данных лиц"""
    try:
        if not os.path.exists(DB_FILE):
            print(f"[DB] Файл базы не найден по пути: {DB_FILE}")
            return {}

        with open(DB_FILE, "r") as f:
            db = json.load(f)
            if not isinstance(db, dict):
                raise ValueError("Формат базы неверный - должен быть словарь")
            print(f"[DB] Загружено {len(db)} пользователей")
            return db

    except json.JSONDecodeError:
        print("[DB ERROR] Файл базы поврежден!")
        return {}
    except Exception as e:
        print(f"[DB ERROR] Ошибка загрузки: {str(e)}")
        return {}


def save_db(db):
    """Сохраняет базу данных с проверками"""
    if not isinstance(db, dict):
        print("[DB ERROR] База должна быть словарем!")
        return False

    try:
        # Создаем папку если ее нет
        os.makedirs(DB_DIR, exist_ok=True)

        # Сохраняем во временный файл
        temp_file = DB_FILE + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)

        # Проверяем что записалось правильно
        with open(temp_file, "r") as f:
            saved = json.load(f)
            if saved != db:
                raise ValueError("Ошибка проверки данных")

        # Заменяем старый файл
        if os.path.exists(DB_FILE):
            os.replace(DB_FILE, DB_FILE + ".bak")
        os.replace(temp_file, DB_FILE)

        print(f"[DB] Успешно сохранено {len(db)} пользователей в {DB_FILE}")
        return True

    except Exception as e:
        print(f"[DB ERROR] Ошибка сохранения: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False