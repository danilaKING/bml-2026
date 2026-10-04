from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent
PYTHON_FILES = [
    PROJECT_DIR / "simple-ex.py",
    PROJECT_DIR / "teach-test.py",
    PROJECT_DIR / "many-params.py",
]


for python_file in PYTHON_FILES:
    print(f"\n=== Запуск {python_file.name} ===")
    subprocess.run([sys.executable, str(python_file)], check=True)

print("\nВсе файлы выполнены.")
