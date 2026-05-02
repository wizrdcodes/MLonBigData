from pathlib import Path
import base64
import html
import re
import subprocess
import sys
import runpy
import io
import contextlib

# Change week number - remember to put plots in outputs
DEFAULT_WEEK = 8
STUDENT_HTML_NAME = "09_AlexanderMurray"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / ("src/mlonbigdata")
SRC_FILES = [
    SRC_DIR / "paths.py",
    SRC_DIR / "plotting.py",
    SRC_DIR / "spark_utils.py",
]

def img_to_base64_data_uri(p: Path) -> str:
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"

def clean_stdout(text: str) -> str:
    # Remove repeated PySpark BrokenPipe traceback blocks first (multi-line)
    text = re.sub(
        r'Traceback \(most recent call last\):.*?BrokenPipeError: \[Errno 32\] Broken pipe\s*',
        '',
        text,
        flags=re.DOTALL
    )

    # Remove common Spark noise line-by-line
    patterns = [
        r"^WARNING: Using incubator modules:.*$",
        r"^Using Spark's default log4j profile:.*$",
        r"^Setting default log level to .*?$",
        r"^To adjust logging level use .*?$",
        r"^.*\bWARN\b.*$",
        r"^\[Stage\s+\d+.*$",
        r"^\s*$",
    ]

    out_lines = []
    for line in text.splitlines():
        if any(re.match(p, line) for p in patterns):
            continue
        out_lines.append(line)

    # Collapse extra blank lines
    cleaned = "\n".join(out_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def _read_text_or_placeholder(p: Path) -> str:
    if p.exists():
        return p.read_text(encoding="utf-8")
    return f"(File not found: {p.as_posix()})"

def _run_python_file(py_file: Path, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, py_file.as_posix()],
        capture_output=True,
        text=True,
    )
    cleaned_output = clean_stdout((proc.stdout or "") + (proc.stderr or ""))
    out_file.write_text(cleaned_output, encoding="utf-8")

def make_html_file(week: int, run_script: bool = True) -> Path:
    code_file = PROJECT_ROOT / f"subs/week{week}/Week{week}Portfolio.py"
    out_html = PROJECT_ROOT / f"subs/week{week}/Week{week}_{STUDENT_HTML_NAME}.html"
    output_file = PROJECT_ROOT / f"subs/week{week}/run_output.txt"
    plots_dir = PROJECT_ROOT / f"outputs/week{week}/plots"

    if run_script:
        _run_python_file(code_file, output_file)

    code_text = _read_text_or_placeholder(code_file)
    raw_out = output_file.read_text(encoding="utf-8") if output_file.exists() else "(run_output.txt not found)"
    out_text = clean_stdout(raw_out)

    plot_files = sorted(plots_dir.glob("*.png"))
    if plot_files:
        plots_html = "".join(
            f"<h3>{html.escape(p.name)}</h3>"
            f"<img src='{img_to_base64_data_uri(p)}' style='max-width:100%;height:auto;border:1px solid #ddd;margin:8px 0 24px 0;' />"
            for p in plot_files
        )
    else:
        plots_html = f"<p>(No .png plots found in {html.escape(plots_dir.as_posix())})</p>"

    existing_src_files = [p for p in SRC_FILES if p.exists()]
    if existing_src_files:
        helpers_html = "<h2>Helper Modules (src/mlonbigdata)</h2>" + "".join(
            f"<h3>{html.escape(p.as_posix())}</h3>"
            f"<pre><code>{html.escape(p.read_text(encoding='utf-8'))}</code></pre>"
            for p in existing_src_files
        )
    else:
        helpers_html = "<h2>Helper Modules (src/mlonbigdata)</h2><p>(No helper .py files found to embed.)</p>"

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Week {week} Portfolio Submission</title>
  <style>
    body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, Arial; margin: 24px; }}
    pre {{ background: #f6f8fa; padding: 12px; overflow-x: auto; border: 1px solid #ddd; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New"; }}
    h2 {{ margin-top: 28px; }}
  </style>
</head>
<body>
  <h1>Week {week} Portfolio Exercise</h1>

  <h2>Code</h2>
  <pre><code>{html.escape(code_text)}</code></pre>

  {helpers_html}

  <h2>Outputs (stdout)</h2>
  <pre><code>{html.escape(out_text)}</code></pre>

  <h2>Plots</h2>
  {plots_html}

</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_doc, encoding="utf-8")
    print(f"[saved] {out_html.resolve()}")
    return out_html

# Call using this function to generate the HTML file
def main():
    make_html_file(DEFAULT_WEEK, run_script=True)

if __name__ == "__main__":
    main()
