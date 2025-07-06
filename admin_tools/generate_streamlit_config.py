import os

base_path = os.path.dirname(__file__)
streamlit_dir = os.path.join(base_path, ".streamlit")
os.makedirs(streamlit_dir, exist_ok=True)

config_path = os.path.join(streamlit_dir, "config.toml")

config_content = """[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#000000"
font = "sans serif"

[server]
headless = true
enableCORS = false
port = 8501
"""

with open(config_path, "w", encoding="utf-8") as f:
    f.write(config_content)

print(f" .streamlit/config.toml generated at: {config_path}")
