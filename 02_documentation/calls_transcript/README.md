# Call Transcripts

Автоматически скачанные транскрипты звонков из Fireflies.ai.

## Структура

```
calls_transcript/
├── download_calls.py          # Скрипт для скачивания
├── README.md                  # Этот файл
├── .gitignore                 # Исключает транскрипты из Git
└── YYYY-MM-DD/               # Папки по датам
    ├── YYYYMMDD_HHMM_meeting-id.md    # Транскрипт (markdown)
    └── YYYYMMDD_HHMM_meeting-id.json  # Полные данные (JSON)
```

## Использование

### Скачать сегодняшние звонки:

```bash
cd calls_transcript
python3 download_calls.py
```

### Требования:

1. **API ключ Fireflies:**
   - Установить переменную окружения: `export FIREFLIES_API_KEY="your_key"`
   - Или добавить в `.env` файл в `chrome-extension-tcs` проекте

2. **Зависимости:**
   - Проект `chrome-extension-tcs` должен быть в `~/projects/chrome-extension-tcs`
   - FirefliesClient должен быть доступен

## Формат файлов

### Markdown (.md)
- Заголовок с названием встречи
- Метаданные (дата, участники, длительность)
- Summary (overview, keywords, action items)
- Полный транскрипт по спикерам

### JSON (.json)
- Полные данные из Fireflies API
- Включает timestamps, sentences, speaker_ids
- Используется для программной обработки

## Конфиденциальность

⚠️ **НЕ коммитить эту папку в Git!**

Транскрипты содержат:
- Личные email адреса
- Внутренние обсуждения
- Action items с именами

Убедись, что `calls_transcript/` в `.gitignore`.

## Организация по датам

Скрипт автоматически сохраняет звонки в папки по датам.
Формат папки: `YYYY-MM-DD` (например, `2025-10-12`)

Это упрощает:
- Поиск звонков по дате
- Архивацию старых звонков
- Очистку временных файлов
