from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../src/mlonbigdata -> project root

def outputs_week_plots(week: int) -> str:
    return (PROJECT_ROOT / "outputs" / f"week{week}" / "plots").as_posix()

