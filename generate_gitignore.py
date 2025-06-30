import os

base_path = os.path.dirname(__file__)
gitignore_path = os.path.join(base_path, ".gitignore")

lines = [
    "# Python",
    "__pycache__/",
    "*.py[cod]",
    "*.pyo",
    "*.pyd",
    "*.swp",
    "",
    "# Jupyter",
    ".ipynb_checkpoints",
    "",
    "# VSCode",
    ".vscode/",
    "",
    "# Environments",
    ".env",
    ".venv",
    "env/",
    "venv/",
    "ENV/",
    "",
    "# OS-specific",
    ".DS_Store",
    "Thumbs.db",
    "",
    "# Streamlit",
    ".streamlit/",
    "",
    "# Exports",
    "exports/",
    "",
    "# Logs",
    "*.log"
]

with open(gitignore_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"✅ .gitignore generated at: {gitignore_path}")
