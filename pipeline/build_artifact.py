"""Package docs/ as a claude.ai artifact folder (same site, without the <html>/<head> wrapper).

    python pipeline/build_artifact.py OUT_DIR
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1])
if out.exists():
    shutil.rmtree(out)
shutil.copytree(ROOT / "docs" / "assets", out / "assets")
shutil.copytree(ROOT / "docs" / "data", out / "data")
h = (ROOT / "docs" / "index.html").read_text()
head = h[h.index("<head>") + 6:h.index("</head>")]
body = h[h.index("<body>") + 6:h.index("</body>")]
for pat in (r'<meta charset[^>]*>\s*', r'<meta name="viewport"[^>]*>\s*', r'<link rel="icon"[^>]*>\s*'):
    head = re.sub(pat, "", head)
head = head.replace("<title>Hatch's Picks</title>",
                    "<title>Hatch's Picks</title>\n<meta name=\"claude:viewport-fit-sides\" content=\"cover\">")
(out / "index.html").write_text(head + body)
print("artifact folder ->", out)
