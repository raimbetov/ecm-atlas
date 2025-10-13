# Task: Unified Multi-Tissue ECM Aging Dashboard

## Цель
Создать интерактивный web-дашборд для визуализации и сравнения экспрессии ECM белков между различными тканями, компартментами и исследованиями в едином интерфейсе.

## Входные данные

**Источник:** `../08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`

**Структура данных:**
- **1,451 записей** (protein × compartment пары)
- **498 уникальных белков**
- **419 белков** встречаются в **2+ компартментах** (84%)
- **2 органа:** Kidney, Intervertebral_disc
- **5 компартментов:** Glomerular, Tubulointerstitial, NP, IAF, OAF
- **2 исследования:** Randles_2021, Tam_2020

## Архитектура Dashboard

### Backend: Flask REST API
- **Endpoint:** `/api/proteins` - список всех белков с метаданными
- **Endpoint:** `/api/heatmap` - данные для heatmap визуализации
- **Endpoint:** `/api/protein/<id>` - детальные данные по белку
- **Endpoint:** `/api/filters` - доступные фильтры (органы, категории, и т.д.)
- **Endpoint:** `/api/stats` - общая статистика

### Frontend: Dynamic HTML + Plotly.js
- Легкий HTML файл (~30-50 KB)
- Динамическая загрузка данных через API
- Интерактивные фильтры
- Responsive дизайн

### Data Storage
- CSV файл как источник данных
- Pandas DataFrame в памяти API server
- Кэширование часто запрашиваемых данных

## Основные визуализации

### 1. **Multi-Compartment Heatmap** (главная визуализация)

**Концепция:**
```
                  Glomerular | Tubulointerstitial | NP    | IAF   | OAF
                  ------------|--------------------| ------|-------|------
COL1A1 (Collagen) |  +2.3     | +1.8              | +0.5  | +1.2  | +0.8
FN1 (Fibronectin) |  -1.1     | -0.9              | +2.1  | +1.5  | +1.8
SERPINA1          |  -0.4     | -0.6              | (нет) | (нет) | (нет)
...
```

**Особенности:**
- **Строки:** Белки (Gene_Symbol + Protein_ID)
- **Колонки:** 5 компартментов
- **Цвет:** Z-score Delta (Old - Young)
  - Синий: снижение с возрастом (-3 до 0)
  - Белый: нет изменений (0)
  - Красный: увеличение с возрастом (0 до +3)
- **Пустые ячейки:** Белок не обнаружен в этом компартменте
- **Hover:** Показывает:
  - Gene Symbol + Protein Name
  - Zscore_Young, Zscore_Old, Zscore_Delta
  - Matrisome Category
  - Dataset Name

**Сортировка:**
- По общей величине изменения (sum of abs(Zscore_Delta))
- По категории матрисомы
- По названию белка (алфавитный)
- По специфичности компартмента

### 2. **Interactive Filters Panel** (боковая панель)

**Фильтры:**

**По органу:**
- [ ] Kidney (458 белков)
- [ ] Intervertebral_disc (993 белков)

**По компартменту:**
- [ ] Glomerular (229)
- [ ] Tubulointerstitial (229)
- [ ] NP (300)
- [ ] IAF (317)
- [ ] OAF (376)

**По категории матрисомы:**
- [ ] Collagens (154)
- [ ] ECM Glycoproteins (453)
- [ ] Proteoglycans (85)
- [ ] ECM Regulators (375)
- [ ] Secreted Factors (222)
- [ ] ECM-affiliated Proteins (162)

**По исследованию:**
- [ ] Randles_2021 (458)
- [ ] Tam_2020 (993)

**По типу изменения:**
- [ ] Increased with aging (Zscore_Delta > 0.5)
- [ ] Decreased with aging (Zscore_Delta < -0.5)
- [ ] Stable (|Zscore_Delta| < 0.5)

**Поиск по белку:**
- Текстовое поле для поиска по Gene_Symbol или Protein_ID
- Autocomplete с предложениями

### 3. **Protein Detail Panel** (при клике на белок)

**Показывает:**
- **Общая информация:**
  - Protein Name
  - Gene Symbol (Canonical)
  - Protein ID (UniProt)
  - Matrisome Category + Division

- **Таблица по компартментам:**
  ```
  Compartment     | Zscore Young | Zscore Old | Delta  | Trend
  ----------------|--------------|------------|--------|--------
  Glomerular      | -0.5         | +1.2       | +1.7   | ↑ Up
  Tubulointerstitial | -0.3      | +0.8       | +1.1   | ↑ Up
  NP              | +0.2         | -0.5       | -0.7   | ↓ Down
  ```

- **Bar chart:** Zscore_Delta для всех компартментов
- **Line chart:** Zscore_Young vs Zscore_Old

### 4. **Summary Statistics Dashboard** (верхняя панель)

**Карточки:**
- Total Proteins Displayed: {N}
- Organs: {N}
- Compartments: {N}
- Avg Zscore Delta: {mean}
- Proteins with multi-compartment presence: {N}

### 5. **Compartment Comparison View** (дополнительная вкладка)

**Scatter plots:**
- Glomerular vs Tubulointerstitial (Kidney compartments)
- NP vs IAF (Disc compartments)
- Kidney vs Disc (cross-organ)

**Показывает:**
- Корреляцию изменений между компартментами
- Outliers (белки с противоположными трендами)

## UI/UX Требования

### Интерактивность
1. **Фильтры применяются мгновенно** без перезагрузки страницы
2. **Heatmap обновляется динамически** при изменении фильтров
3. **Клик на белок** → открывает detail panel справа
4. **Hover на ячейку** → tooltip с подробной информацией
5. **Экспорт данных:**
   - CSV текущей выборки
   - PNG скриншот heatmap
   - PDF отчет по выбранным белкам

### Адаптивный дизайн
- **Desktop:** 3-колонка (filters | heatmap | details)
- **Tablet:** 2-колонка (filters collapse | heatmap)
- **Mobile:** Вертикальный stack

### Производительность
- Lazy loading для больших heatmap (виртуализация)
- Debounced search (задержка 300ms)
- Кэширование API запросов в браузере
- Loading indicators для всех async операций

## Технический стек

### Backend
```python
flask==3.1.2
flask-cors==6.0.1
pandas==2.2.0
numpy==1.26.0
```

### Frontend
```html
plotly.js 2.26.0 (CDN)
chart.js 4.0.0 (для дополнительных графиков)
vanilla JavaScript (no frameworks)
CSS Grid + Flexbox (responsive)
```

## Структура файлов

```
09_unified_dashboard/
├── 00_TASK_UNIFIED_DASHBOARD.md      # Эта спецификация
├── api_server.py                      # Flask API
├── dashboard.html                     # Главная страница
├── static/
│   ├── styles.css                     # Стили
│   └── app.js                         # JavaScript логика
├── 00_start_servers.sh                # Скрипт запуска API + HTTP
└── screenshots/                        # Скриншоты для валидации
```

## API Endpoints Specification

### GET `/api/proteins`
**Response:**
```json
{
  "proteins": [
    {
      "protein_id": "P02452",
      "gene_symbol": "COL1A1",
      "protein_name": "Collagen alpha-1(I) chain",
      "matrisome_category": "Collagens",
      "compartments": ["Glomerular", "Tubulointerstitial", "NP", "IAF", "OAF"]
    },
    ...
  ]
}
```

### GET `/api/heatmap?organs=Kidney&categories=Collagens`
**Query params:**
- `organs` (optional): comma-separated list
- `compartments` (optional): comma-separated list
- `categories` (optional): comma-separated list
- `studies` (optional): comma-separated list
- `trend` (optional): "up", "down", "stable"
- `search` (optional): search query for gene/protein

**Response:**
```json
{
  "proteins": ["COL1A1", "COL1A2", ...],
  "compartments": ["Glomerular", "Tubulointerstitial", "NP", "IAF", "OAF"],
  "data": {
    "COL1A1": {
      "Glomerular": {"zscore_delta": 1.5, "zscore_young": -0.5, "zscore_old": 1.0},
      "Tubulointerstitial": {"zscore_delta": 1.2, "zscore_young": -0.3, "zscore_old": 0.9},
      "NP": null,
      "IAF": null,
      "OAF": null
    },
    ...
  },
  "metadata": {
    "total_proteins": 150,
    "avg_zscore_delta": 0.8
  }
}
```

### GET `/api/protein/<protein_id>`
**Response:**
```json
{
  "protein_id": "P02452",
  "gene_symbol": "COL1A1",
  "protein_name": "Collagen alpha-1(I) chain",
  "matrisome_category": "Collagens",
  "matrisome_division": "Core matrisome",
  "compartments": [
    {
      "organ": "Kidney",
      "compartment": "Glomerular",
      "dataset": "Randles_2021",
      "zscore_young": -0.5,
      "zscore_old": 1.0,
      "zscore_delta": 1.5,
      "abundance_young": 1000.5,
      "abundance_old": 1500.2
    },
    ...
  ]
}
```

### GET `/api/filters`
**Response:**
```json
{
  "organs": [
    {"name": "Kidney", "count": 458},
    {"name": "Intervertebral_disc", "count": 993}
  ],
  "compartments": [
    {"name": "Glomerular", "count": 229, "organ": "Kidney"},
    ...
  ],
  "categories": [
    {"name": "Collagens", "count": 154},
    ...
  ],
  "studies": [
    {"name": "Randles_2021", "count": 458},
    {"name": "Tam_2020", "count": 993}
  ]
}
```

### GET `/api/stats`
**Response:**
```json
{
  "total_proteins": 498,
  "total_entries": 1451,
  "multi_compartment_proteins": 419,
  "organs": 2,
  "compartments": 5,
  "studies": 2,
  "avg_zscore_delta": 0.127,
  "std_zscore_delta": 0.712
}
```

## Критерии успеха

### Функциональные
✅ Все 498 белков доступны для просмотра
✅ Heatmap показывает все 5 компартментов
✅ Фильтры работают корректно и мгновенно
✅ Поиск по белкам работает с autocomplete
✅ Detail panel открывается при клике на белок
✅ Цветовая шкала корректно отражает z-scores
✅ Пустые ячейки (белок отсутствует) четко видны
✅ Экспорт данных работает (CSV, PNG)

### Производительность
✅ Загрузка initial view < 2 секунд
✅ Фильтрация < 500ms
✅ Hover tooltip < 100ms
✅ Heatmap с 100+ белками рендерится плавно

### UX
✅ Интуитивный интерфейс (не требует инструкций)
✅ Responsive на всех устройствах
✅ Доступность (ARIA labels, keyboard navigation)
✅ Четкие loading states

## Вопросы для согласования

### 1. Приоритет визуализаций
**Вопрос:** Какая визуализация самая важная?
- **Вариант A:** Heatmap (protein × compartment) - показывает все данные сразу
- **Вариант B:** Protein detail view - фокус на отдельных белках
- **Вариант C:** Compartment comparison - кросс-компартментный анализ

**Рекомендация:** Начать с Heatmap как главной визуализации, остальное добавить позже.

### 2. Фильтрация vs Выборка
**Вопрос:** Как работают фильтры?
- **Вариант A:** AND логика (все условия должны выполняться)
- **Вариант B:** OR логика внутри категории (Kidney OR Disc), AND между категориями
- **Вариант C:** Пользователь выбирает логику

**Рекомендация:** Вариант B - наиболее гибкий.

### 3. Обработка missing values
**Вопрос:** Как показывать белки, которые отсутствуют в некоторых компартментах?
- **Вариант A:** Пустая ячейка (белая)
- **Вариант B:** Серая ячейка с текстом "N/A"
- **Вариант C:** Скрывать такие белки из heatmap

**Рекомендация:** Вариант B - наглядно показывает отсутствие данных.

### 4. Масштабирование
**Вопрос:** Как добавлять новые исследования?
- **Вариант A:** Ручное обновление CSV → перезапуск API
- **Вариант B:** Автоматический мониторинг CSV файла
- **Вариант C:** Отдельный endpoint для загрузки новых данных

**Рекомендация:** Вариант A для начала (простота), Вариант C для production.

### 5. Z-score color scale
**Вопрос:** Какой диапазон для цветовой шкалы?
- **Вариант A:** Fixed [-3, +3] (стандартно для z-scores)
- **Вариант B:** Dynamic по данным (min/max из выборки)
- **Вариант C:** User-adjustable (слайдер)

**Рекомендация:** Вариант A с возможностью переключения на Вариант C.

### 6. Deployment
**Вопрос:** Где разместить dashboard?
- **Вариант A:** Локальный запуск (localhost)
- **Вариант B:** GitHub Pages (статика + API на Vercel/Render)
- **Вариант C:** Docker container

**Рекомендация:** Вариант A для разработки, документировать для Вариант B в будущем.

## Next Steps

После согласования спецификации:

1. **Phase 1:** Backend API (api_server.py)
   - Implement all endpoints
   - Test with sample queries
   - Document API with examples

2. **Phase 2:** Frontend Core (dashboard.html)
   - Main heatmap visualization
   - Basic filters (organ, compartment)
   - Search functionality

3. **Phase 3:** Enhanced UI
   - Protein detail panel
   - Advanced filters
   - Export functionality

4. **Phase 4:** Testing & Optimization
   - Headless browser screenshots
   - Performance profiling
   - Bug fixes

5. **Phase 5:** Documentation
   - User guide
   - API documentation
   - Deployment instructions
