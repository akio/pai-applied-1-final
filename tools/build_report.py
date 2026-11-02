"""Markdown で書いた資料を HTML にし、Chrome (ヘッドレス) で PDF に印刷する。

    python tools/build_report.py                                  # docs/go2_sprawling_report.md -> .html / .pdf
    python tools/build_report.py docs/other.md --css docs/report.css

Markdown の書き方 (Python-Markdown の tables / attr_list / md_in_html を使う):
  - 画像だけの段落 ![説明](img/a.jpg) は、説明を図の下のキャプションにした図になる
  - 段落の次の行に {: .sub } / {: .note } と書くとクラスが付く (見た目は CSS で決める)
  - 1 行に \\newpage だけを書くと、そこで改ページする
  - 横に並べるときは <div class="row" markdown="1"><div class="col" markdown="1"> ... </div></div>
HTML は Markdown と同じフォルダに作るので、画像のパスは Markdown のファイルからの相対パスで書く。
"""

import argparse
import html
import os
import re
import shutil
import subprocess

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def to_html(md_text, css):
    md_text = re.sub(r"^\\newpage\s*$", '<div class="page-break"></div>', md_text, flags=re.MULTILINE)
    body = markdown.markdown(md_text, extensions=["tables", "attr_list", "md_in_html"])
    # 画像だけの段落を、alt をキャプションにした figure にする
    body = re.sub(
        r'<p><img alt="([^"]*)" src="([^"]*)" ?/?></p>',
        lambda m: f'<figure><img src="{m.group(2)}"><figcaption>{m.group(1)}</figcaption></figure>',
        body,
    )
    title = re.search(r"<h1[^>]*>(.*?)</h1>", body)
    title = html.unescape(re.sub(r"<[^>]+>", "", title.group(1))) if title else "report"
    return TEMPLATE.format(title=html.escape(title), css=css, body=body)


def find_chrome():
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    raise SystemExit("Chrome / Chromium が見つからない")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("markdown", nargs="?", default=os.path.join(ROOT, "docs", "go2_sprawling_report.md"))
    parser.add_argument("--css", default=os.path.join(ROOT, "docs", "report.css"))
    parser.add_argument("--html-only", action="store_true", help="HTML だけ作る (PDF にしない)")
    args = parser.parse_args()

    md_path = os.path.abspath(args.markdown)
    stem = os.path.splitext(md_path)[0]
    with open(md_path, encoding="utf-8") as f:
        md_text = f.read()
    with open(args.css, encoding="utf-8") as f:
        css = f.read()
    with open(stem + ".html", "w", encoding="utf-8") as f:
        f.write(to_html(md_text, css))
    print(f"saved {stem}.html")
    if args.html_only:
        return

    subprocess.run(
        [find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={stem}.pdf", f"file://{stem}.html"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    pages = ""
    if shutil.which("pdfinfo"):
        info = subprocess.run(["pdfinfo", f"{stem}.pdf"], capture_output=True, text=True).stdout
        pages = re.search(r"Pages:\s+(\d+)", info)
        pages = f" ({pages.group(1)} pages)" if pages else ""
    print(f"saved {stem}.pdf{pages}")


if __name__ == "__main__":
    main()
