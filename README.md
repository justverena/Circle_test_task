# Fine-Tuning Mistral-7B с QLoRA

Репозиторий демонстрирует процесс **instruction fine-tuning** модели **Mistral-7B-Instruct** с использованием **QLoRA** и **LoRA adapters** на небольшом наборе данных.

## Структура

- `notebooks/01_data_preparation.ipynb` — подготовка и очистка датасета.  
- `notebooks/02_finetuning.ipynb` — fine-tuning модели с использованием LoRA.  
- `notebooks/03_evaluation.ipynb` — генерация ответов и оценка качества модели (ROUGE).  
- `data/dataset.jsonl` — финальный подготовленный датасет для обучения (300 примеров).  
- `mistral-lora-adapter/` — сохранённый fine-tuned LoRA adapter.  


## 1. Подготовка данных

Использован датасет **[databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)** с Hugging Face.

Основные шаги:

1. Загрузка датасета.  
2. Очистка данных: удаление пустых инструкций/ответов, фильтрация по длине ответа (10–300 слов).  
3. Удаление дубликатов.  
4. Ограничение количества примеров до 300.  
5. Добавление префикса к инструкциям `"Answer clearly and concisely:"`.  
6. Сохранение в `JSONL` для дальнейшего fine-tuning.

Пример записи датасета:
```
{
  "instruction": "Answer clearly and concisely: When did Virgin Australia start operating?",
  "response": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route."
}
```
## 2. FineTuning модели

Использован **QLoRA** для тонкой настройки модели **Mistral-7B-Instruct**:

- Загрузка токенизатора и подготовка данных.  
- Форматирование данных в пары **Instruction - Response**.  
- Токенизация и обрезка до 512 токенов.  
- Настройка LoRA:
  - `r=16`, `lora_alpha=32`, `target_modules=["q_proj", "v_proj"]`, `dropout=0.05`.  
- Обучение модели 1 эпоху с логированием в **Weights & Biases** (`run_name="mistral-lora-finetune"`).  
- Сохранение адаптера: `mistral-lora-adapter/`.  

Пример генерации ответа после fine-tuning:
```
response = ft_model.generate(
    tokenizer("Explain machine learning in simple terms", return_tensors="pt").to("cuda"),
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7
)
```
## W&B graphs

- Все тренировки логировались в **Weights & Biases**.  
- Ссылка на run: [W&B](https://wandb.ai/v_gaaze-/huggingface/runs/uybg09e9)


## 3. Evaluation

Проводится сравнение **базовой модели** и **fine tuned модели** на 10 новых инструкциях.

- Генерация ответов обеими моделями.  
- Сравнение с помощью метрики **ROUGE**.

Пример prompts и ответов:

| Prompt | Base Model | Fine Tuned Model |
|--------|------------|-----------------|
| Explain what machine learning is in simple terms. | Machine learning is a process that allows computers to learn from data... | Machine learning is a type of AI where a computer teaches and learns on its own... |
| What are the benefits of regular exercise? | Regular exercise has numerous benefits, including: improved health, sleep, energy... | Regular exercise can help reduce chronic disease risk, improve mental health... |

оценка ROUGE (Fine tuned vs Base Model):
| Metric      | Score  |
|-------------|--------|
| ROUGE-1     | 0.4987 |
| ROUGE-2     | 0.2435 |
| ROUGE-L     | 0.3442 |
| ROUGE-Lsum  | 0.3839 |

## 4. Выводы
1.	**Ясность и краткость**: Fine tuned модель даёт более короткие, понятные ответы, избегает лишней информации.
2.	**Детализация**: Базовая модель иногда генерирует более длинные и подробные ответы, тогда как fine tuned модель сокращает детали, делая ответы более сжатыми.
3.	**Последовательность в терминологии**: Fine tuned модель использует более последовательные формулировки под стиль “Answer clearly and concisely”.
4.	**ROUGE**: Средние значения ROUGE отражают умеренное лексическое совпадение (около 0.3–0.5), что ожидаемо, так как fine tuned модель перефразирует ответы, сохраняя смысл.


Вывод: Fine tuning с LoRA улучшил ясность и структуру ответов, сохранив знания базовой модели. Модель стала лучше подходить для генерации коротких и понятных инструкций без потери содержания.

## 5. Как использовать

1. Клонировать репозиторий:

git clone `https://github.com/justverena/Circle_test_task.git`
cd `Circle_test_task`

2.	Подготовить окружение:

pip install -r requirements.txt

3.	Запустить ноутбуки по порядку:
	•	01_data_preparation.ipynb — подготовка датасета
	•	02_finetuning.ipynb — fine-tuning
	•	03_evaluation.ipynb — генерация и evaluation