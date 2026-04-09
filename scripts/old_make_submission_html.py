# from pathlib import Path
# import base64
# import html
# import re
#
# # --- EDIT THESE THREE LINES --- Weeks += 1
# CODE_FILE = Path("subs/week3/Week3Portfolio.py")
# OUT_HTML = Path("../subs/week3/Week3_09_AlexanderMurray.html")  # MUST match their exact naming format
# OUTPUT_FILE = Path("../subs/week3/run_output.txt")
#
# PLOTS_DIR = Path("../plots")  # MAKE SURE TO HAVE PLOTS FOLDER ADJACENT
#
# # CHANGE TITLE DOWN BELOW (2 PLACES: <title>, <h1>)
#
# # RUN IN TERMINAL AFTER CREATING WEEK# FOLDER IN /submissions, REPLACE #S:
# # python Week#Portfolio.py > submissions/week#/run_output.txt 2>&1
#
# # GENERATE HTML:
# # python make_submission_html.py
#
# def img_to_base64_data_uri(p: Path) -> str:
#     data = p.read_bytes()
#     b64 = base64.b64encode(data).decode("ascii")
#     # assuming png; if you saved jpg, change image/png accordingly
#     return f"data:image/png;base64,{b64}"
#
# def clean_stdout(text: str) -> str:
#     """
#     Remove Spark/Java startup warnings + Spark stage progress lines.
#     """
#     patterns = [
#         r"^WARNING: Using incubator modules:.*$",
#         r"^Using Spark's default log4j profile:.*$",
#         r"^Setting default log level to .*?$",
#         r"^To adjust logging level use .*?$",
#         r"^.*\bWARN\b.*$",  # Spark WARN lines (Utils, NativeCodeLoader, SparkStringUtils, etc.)
#         r"^\[Stage\s+\d+.*$",  # [Stage 1:> ...]
#         r"^\s*$",  # optional: remove blank lines (comment out if you want blanks kept)
#     ]
#
#     out_lines = []
#     for line in text.splitlines():
#         if any(re.match(p, line) for p in patterns):
#             continue
#         out_lines.append(line)
#     return "\n".join(out_lines).strip()
#
# def main():
#     code_text = CODE_FILE.read_text(encoding="utf-8")
#     raw_out = OUTPUT_FILE.read_text(encoding="utf-8") if OUTPUT_FILE.exists() else "(run_output.txt not found)"
#     out_text = clean_stdout(raw_out)
#
#     plot_files = sorted(PLOTS_DIR.glob("*.png"))
#
#     plots_html = ""
#     if plot_files:
#         for p in plot_files:
#             uri = img_to_base64_data_uri(p)
#             plots_html += (
#                 f"<h3>{html.escape(p.name)}</h3>"
#                 f"<img src='{uri}' style='max-width:100%;height:auto;border:1px solid #ddd;margin:8px 0 24px 0;' />"
#             )
#     else:
#         plots_html = "<p>(No .png plots found in ./plots)</p>"
#
#     # NOTE: Code first now
#     # CHANGE TITLE WEEK # ----------
#     html_doc = f"""<!doctype html>
# <html>
# <head>
#   <meta charset="utf-8"/>
#   <title>Week 3 Portfolio Submission</title>
#   <style>
#     body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, Arial; margin: 24px; }}
#     pre {{ background: #f6f8fa; padding: 12px; overflow-x: auto; border: 1px solid #ddd; }}
#     code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New"; }}
#     h2 {{ margin-top: 28px; }}
#   </style>
# </head>
# <body>
#   <h1>Week 3 Portfolio Exercise</h1>
#
#   <h2>Code</h2>
#   <pre><code>{html.escape(code_text)}</code></pre>
#
#   <h2>Outputs (stdout)</h2>
#   <pre><code>{html.escape(out_text)}</code></pre>
#
#   <h2>Plots</h2>
#   {plots_html}
#
# </body>
# </html>
# """
#     OUT_HTML.write_text(html_doc, encoding="utf-8")
#     print(f"[saved] {OUT_HTML.resolve()}")
#
#
# if __name__ == "__main__":
#     main()