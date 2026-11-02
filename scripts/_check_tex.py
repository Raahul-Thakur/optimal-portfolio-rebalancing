import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tex = open(os.path.join(ROOT, "paper", "main.tex"), encoding="utf-8").read()
bib = open(os.path.join(ROOT, "paper", "refs.bib"), encoding="utf-8").read()

begins = re.findall(r"\\begin\{(\w+\*?)\}", tex)
ends = re.findall(r"\\end\{(\w+\*?)\}", tex)
cb, ce = Counter(begins), Counter(ends)
bad = [k for k in set(list(cb) + list(ce)) if cb[k] != ce[k]]
print("env begins", len(begins), "ends", len(ends), "| unbalanced:", bad or "none")

figs = re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
print("figures referenced:", len(figs))
for f in figs:
    p = os.path.join(ROOT, "results", "figures", f)
    print("   ", "OK " if os.path.exists(p) else "MISSING", f)

cites = set(re.findall(r"\\cite[tp]?\{([^}]+)\}", tex))
keys = set()
for c in cites:
    keys.update(x.strip() for x in c.split(","))
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
print("citation keys used:", len(keys),
      "| missing from bib:", (keys - bibkeys) or "none")
print("bib entries:", len(bibkeys), "| unused:", (bibkeys - keys) or "none")
