"""One-off: build a local .streamlit/secrets.toml from .env + holdings.csv.

Gitignored output; mirrors what you paste into the Streamlit Cloud dashboard.
Never prints secret values.
"""
from pathlib import Path

env = {}
for line in Path(".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    env[k.strip()] = v.strip()

holdings = Path("holdings.csv").read_text(encoding="utf-8").strip()


def toml_str(v: str) -> str:
    v = v.replace(chr(92), chr(92) * 2).replace('"', chr(92) + '"')
    return '"' + v + '"'


keys = ("FMP_API_KEY", "FINNHUB_API_KEY", "ALPHAVANTAGE_API_KEY", "QUIVER_API_KEY", "SEC_USER_AGENT")
lines = [
    "# LOCAL secrets - gitignored, never committed. Mirrors what you paste into",
    "# Streamlit Cloud (App -> Settings -> Secrets). Change APP_PASSWORD below.",
    "",
    f"APP_PASSWORD = {toml_str('rmtest123')}   # <-- CHANGE before/at deploy",
    "",
]
lines += [f"{k} = {toml_str(env[k])}" for k in keys if k in env]
lines += ["", 'HOLDINGS_CSV = """', holdings, '"""']

Path(".streamlit/secrets.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")

print("wrote .streamlit/secrets.toml")
print("  keys:", [k for k in keys if k in env])
print("  APP_PASSWORD: placeholder 'rmtest123' (change before deploy)")
print("  holdings data rows:", max(0, holdings.count(chr(10))))
