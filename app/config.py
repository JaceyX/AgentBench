from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_URL = f"sqlite:///{BASE_DIR / 'agentbench.db'}"

BENCHMARK_PATH = BASE_DIR / "app" / "datasets" / "benchmark.jsonl"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
