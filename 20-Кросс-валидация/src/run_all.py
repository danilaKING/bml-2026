"""Последовательный запуск всех модулей и пересборка каталога figures.

    python run_all.py            все модули
    python run_all.py 01 06      только указанные модули

Полный прогон занимает около двух минут; наибольшее время требуют модули
05_nested_cv.py и 08_choosing_k.py.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

MODULES = [
    ("01_why_cv.py", "Постановка задачи"),
    ("02_python_example.py", "Программная реализация"),
    ("03_pipeline_leakage.py", "Смещение при утечке информации"),
    ("04_model_selection.py", "Подбор гиперпараметров и выбор модели"),
    ("05_nested_cv.py", "Вложенная кросс-валидация"),
    ("06_manual_example.py", "Численный пример"),
    ("07_cv_strategies.py", "Схемы разбиения выборки"),
    ("08_choosing_k.py", "Сравнение схем оценивания"),
]


def main():
    selected = sys.argv[1:]
    started = time.perf_counter()
    failed = []

    for name, title in MODULES:
        if selected and not any(key in name for key in selected):
            continue
        print("\n" + "#" * 74)
        print(f"# {name} — {title}")
        print("#" * 74)
        module_started = time.perf_counter()
        result = subprocess.run([sys.executable, name], cwd=HERE)
        elapsed = time.perf_counter() - module_started
        if result.returncode != 0:
            failed.append(name)
            print(f"{name}: завершено с кодом {result.returncode}")
        else:
            print(f"{name}: выполнено за {elapsed:.1f} с")

    figures = sorted((HERE / "figures").glob("*.png"))
    print("\n" + "=" * 74)
    print(f"Рисунков в каталоге figures: {len(figures)}")
    for path in figures:
        print(f"  {path.name}")
    print(f"Общее время: {time.perf_counter() - started:.1f} с")

    if failed:
        print(f"Модули, завершившиеся с ошибкой: {', '.join(failed)}")
        sys.exit(1)
    print("Все модули выполнены успешно.")


if __name__ == "__main__":
    main()
