import json
import os
from cryptography.fernet import Fernet

# Конфигурация
DB_DIR = os.path.join(os.path.dirname(__file__), '')
DB_FILE = os.path.join(DB_DIR, 'face_db.enc')
KEY_FILE = os.path.join(DB_DIR, '.face_db.key')


class DBEncryptor:
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)

    def _get_or_create_key(self):
        """Получаем или создаем ключ шифрования"""
        if os.path.exists(KEY_FILE):
            with open(KEY_FILE, 'rb') as f:
                return f.read()

        # Генерируем новый ключ на основе системной информации + случайных данных
        key = Fernet.generate_key()
        with open(KEY_FILE, 'wb') as f:
            f.write(key)
        os.chmod(KEY_FILE, 0o600)  # Только для владельца
        return key

    def encrypt_data(self, data):
        """Шифруем данные с добавлением HMAC"""
        json_data = json.dumps(data).encode('utf-8')
        return self.cipher.encrypt(json_data).decode('ascii')

    def decrypt_data(self, encrypted_data):
        """Дешифруем данные с проверкой HMAC"""
        try:
            json_data = self.cipher.decrypt(encrypted_data.encode('ascii'))
            return json.loads(json_data.decode('utf-8'))
        except Exception as e:
            print(f"[DB ERROR] Decryption failed: {str(e)}")
            return {}


def load_db():
    """Загружает и дешифрует базу данных"""
    encryptor = DBEncryptor()

    if not os.path.exists(DB_FILE):
        print(f"[DB] Файл базы не найден. Будет создан новый.")
        return {}

    try:
        with open(DB_FILE, "r") as f:
            encrypted = f.read()
            db = encryptor.decrypt_data(encrypted)

            if not isinstance(db, dict):
                raise ValueError("Invalid database format")

            print(f"[DB] Загружено {len(db)} пользователей")
            return db

    except Exception as e:
        print(f"[DB ERROR] Ошибка загрузки: {str(e)}")
        return {}


def save_db(db):
    """Шифрует и сохраняет базу данных"""
    if not isinstance(db, dict):
        print("[DB ERROR] База должна быть словарем!")
        return False

    encryptor = DBEncryptor()

    try:
        os.makedirs(DB_DIR, exist_ok=True)

        # Шифруем данные
        encrypted = encryptor.encrypt_data(db)

        # Сохраняем во временный файл
        temp_file = DB_FILE + ".tmp"
        with open(temp_file, "w") as f:
            f.write(encrypted)

        # Проверяем что можно расшифровать
        test_data = encryptor.decrypt_data(encrypted)
        if test_data != db:
            raise ValueError("Ошибка проверки данных")

        # Атомарная замена файла
        if os.path.exists(DB_FILE):
            os.replace(DB_FILE, DB_FILE + ".bak")
        os.replace(temp_file, DB_FILE)

        print(f"[DB] Успешно сохранено {len(db)} пользователей")
        return True

    except Exception as e:
        print(f"[DB ERROR] Ошибка сохранения: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False