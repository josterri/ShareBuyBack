import os

base_path = os.path.dirname(__file__)
req_path = os.path.join(base_path, "requirements.txt")

requirements = [
    "streamlit",
    "pandas",
    "numpy",
    "yfinance",
    "fpdf",
    "pytest"
]

with open(req_path, "w", encoding="utf-8") as f:
    f.write("\n".join(requirements) + "\n")

print(f"✅ requirements.txt generated at: {req_path}")
