# Task: ECM Dataset Merge - Объединение данных внеклеточного матрикса

## Цель
Объединить z-score нормализованные данные из нескольких исследований в единый стандартизированный CSV файл для загрузки в интерактивный дашборд.

## Входные данные

### Randles et al. 2021 - Kidney Aging (Homo sapiens)
**Источник:** `06_Randles_z_score_by_tissue_compartment/claude_code/`

1. **Randles_2021_Glomerular_zscore.csv**
   - Ткань: Kidney
   - Компартмент: Glomerular
   - Всего белков: 2,610
   - ECM белков: 229 (8.8%)
   - Метод: Label-free LC-MS/MS (Progenesis + Mascot)

2. **Randles_2021_Tubulointerstitial_zscore.csv**
   - Ткань: Kidney
   - Компартмент: Tubulointerstitial
   - Всего белков: 2,610
   - ECM белков: 229 (8.8%)
   - Метод: Label-free LC-MS/MS (Progenesis + Mascot)

### Tam et al. 2020 - Intervertebral Disc Aging (Homo sapiens)
**Источник:** `07_Tam_2020_paper_to_csv/claude_code/`

3. **Tam_2020_NP_zscore.csv**
   - Ткань: Intervertebral_disc
   - Компартмент: NP (Nucleus Pulposus)
   - Всего белков: 300
   - ECM белков: 300 (100% - уже отфильтровано)
   - Метод: Label-free LC-MS/MS (MaxQuant LFQ)

4. **Tam_2020_IAF_zscore.csv**
   - Ткань: Intervertebral_disc
   - Компартмент: IAF (Inner Annulus Fibrosus)
   - Всего белков: 317
   - ECM белков: 317 (100% - уже отфильтровано)
   - Метод: Label-free LC-MS/MS (MaxQuant LFQ)

5. **Tam_2020_OAF_zscore.csv**
   - Ткань: Intervertebral_disc
   - Компартмент: OAF (Outer Annulus Fibrosus)
   - Всего белков: 376
   - ECM белков: 376 (100% - уже отфильтровано)
   - Метод: Label-free LC-MS/MS (MaxQuant LFQ)

## Схема колонок

### Общие колонки (присутствуют во всех файлах):
- `Protein_ID` - UniProt ID
- `Protein_Name` - Полное название белка
- `Gene_Symbol` - Символ гена
- `Tissue` - Орган/ткань
- `Tissue_Compartment` - Компартмент внутри ткани
- `Species` - Организм (Homo sapiens)
- `Abundance_Young` - Abundance в молодом возрасте
- `Abundance_Old` - Abundance в старом возрасте
- `Zscore_Young` - Z-score для молодого возраста
- `Zscore_Old` - Z-score для старого возраста
- `Zscore_Delta` - Изменение z-score (Old - Young)
- `Method` - Метод протеомики
- `Study_ID` - Идентификатор исследования (Randles_2021 / Tam_2020)
- `Canonical_Gene_Symbol` - Канонический символ гена
- `Matrisome_Category` - Категория матрисомы (ECM Glycoproteins, Collagens, и т.д.)
- `Matrisome_Division` - Раздел матрисомы (Core matrisome / Matrisome-associated)
- `Match_Level` - Уровень совпадения при аннотации
- `Match_Confidence` - Уверенность в совпадении (0-100)

### Специфичные колонки Randles 2021:
- `Abundance_Young_transformed` - log2-трансформированное значение Young
- `Abundance_Old_transformed` - log2-трансформированное значение Old

### Специфичные колонки Tam 2020:
- `N_Profiles_Young` - Количество профилей для молодого возраста
- `N_Profiles_Old` - Количество профилей для старого возраста

## Задачи

### 1. Фильтрация ECM белков
- **Для Randles 2021:** Оставить только строки где `Matrisome_Division` содержит "Core matrisome" или "Matrisome-associated"
- **Для Tam 2020:** Уже отфильтровано (100% ECM белков)

### 2. Добавление метаданных
Добавить новые колонки для отслеживания источника:
- `Dataset_Name` - Название исследования ("Randles_2021" / "Tam_2020")
- `Organ` - Орган ("Kidney" / "Intervertebral_disc")
- `Compartment` - Компартмент ("Glomerular", "Tubulointerstitial", "NP", "IAF", "OAF")

### 3. Стандартизация колонок
- Унифицировать порядок колонок
- Заполнить отсутствующие колонки значениями NaN/пусто:
  - Для Randles: добавить `N_Profiles_Young`, `N_Profiles_Old` (пусто)
  - Для Tam: добавить `Abundance_Young_transformed`, `Abundance_Old_transformed` (пусто)

### 4. Объединение данных
Объединить все 5 CSV файлов в один DataFrame с использованием `pd.concat()`

### 5. Проверка качества
- Убедиться, что нет дубликатов (Protein_ID + Compartment должны быть уникальны)
- Проверить, что все обязательные колонки заполнены
- Проверить диапазоны z-score (-5 до +5)
- Сформировать отчет о статистике:
  - Количество белков по исследованиям
  - Количество белков по тканям
  - Количество белков по компартментам
  - Распределение по категориям матрисомы

## Ожидаемый выход

### Файл: `merged_ecm_aging_zscore.csv`
**Формат:**
```
Dataset_Name,Organ,Compartment,Protein_ID,Protein_Name,Gene_Symbol,...
Randles_2021,Kidney,Glomerular,P02452,Collagen alpha-1(I) chain,COL1A1,...
Randles_2021,Kidney,Tubulointerstitial,P02452,Collagen alpha-1(I) chain,COL1A1,...
Tam_2020,Intervertebral_disc,NP,P02452,Collagen alpha-1(I) chain,COL1A1,...
...
```

**Ожидаемое количество строк:**
- Randles Glomerular: 229 ECM белков
- Randles Tubulointerstitial: 229 ECM белков
- Tam NP: 300 ECM белков
- Tam IAF: 317 ECM белков
- Tam OAF: 376 ECM белков
- **Итого:** ~1,451 строк (без учета дубликатов)

### Файл: `merge_report.md`
Отчет с:
- Статистикой по исследованиям
- Списком уникальных белков
- Распределением по категориям
- Любыми предупреждениями или проблемами

## Критерии успеха

✅ Все 5 входных CSV файлов объединены
✅ Только ECM белки включены (фильтр по Matrisome_Division)
✅ Добавлены метаданные (Dataset_Name, Organ, Compartment)
✅ Все колонки стандартизированы
✅ Нет дубликатов (Protein_ID + Compartment уникальны)
✅ Z-scores в корректном диапазоне
✅ Файл готов для загрузки в дашборд
✅ Сформирован отчет о качестве данных

## Вопросы для уточнения

1. **Дубликаты белков между компартментами:** Один и тот же белок (например, COL1A1) может присутствовать в нескольких компартментах с разными z-scores. Это ожидаемо и нормально?

2. **Обработка отсутствующих значений:** Как заполнять пустые значения в специфичных колонках (Abundance_transformed, N_Profiles)? Оставить пустыми или использовать значение по умолчанию?

3. **Формат выходного файла:** CSV с кодировкой UTF-8, разделитель запятая? Нужна ли сжатая версия (CSV.gz)?

4. **Дашборд интеграция:** Планируется обновить существующий дашборд из `06_Randles_z_score_by_tissue_compartment/` или создать новый multi-tissue дашборд?

5. **Дополнительные исследования:** Планируется ли добавление других исследований из `data_raw/` в будущем? Если да, нужно учесть расширяемость схемы.
