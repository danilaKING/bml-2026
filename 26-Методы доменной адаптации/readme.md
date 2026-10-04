# Методы доменной адаптации

### Команда: 26 Литовченок Дмирий, Суханов Фёдор 5030102/30201

## 1. Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt

python src/coral_numeric_example.py   # ручной пример CORAL по шагам 
python src/run_all.py                 # эксперимент: таблица + графики 
python src/make_figures.py            # поясняющие картинки для слайдов 
```

`make_figures.py` запускается после `run_all.py`: график по seed берёт данные из `results/per_seed.csv`.

Дополнительно:
```bash
python src/run_all.py --sweep       # + график точности от угла поворота (дольше)
python src/run_all.py --angle 50    # другой угол поворота target
```

## 2. Где результаты

| Что | Где |
|---|---|
| Таблица точностей | `results/results.md` |
| Точность по каждому seed | `results/per_seed.csv` |
| Графики эксперимента | `figures/` |
| Картинки для слайдов | `figures/explain/` |
