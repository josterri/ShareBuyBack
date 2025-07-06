import os


# generate_requirements.py

import ast
from pathlib import Path

# A small set of standard-library modules to ignore
STDLIB = {
    'sys', 'os', 'math', 'json', 'ast', 'pathlib', 'datetime', 'time',
    're', 'threading', 'subprocess', 'logging'
}

base_path = os.path.dirname(__file__)
req_path = os.path.join(base_path, "requirements.txt")
def find_imports(py_path):
    """Yield top-level module names imported in the given .py file."""
    with open(py_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=str(py_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                yield node.module.split('.')[0]

def main():
    all_imports = set()
    for py in Path('.').rglob('*.py'):
        for mod in find_imports(py):
            all_imports.add(mod)
    # Filter out stdlib and local modules (assuming local modules are lowercase and match your folder names)
    externals = sorted(
        m for m in all_imports
        if m not in STDLIB and not (Path(f"{m}.py").exists() or Path(m).is_dir())
    )
    # Write requirements.txt
    with open('requirements.txt', 'w', encoding='utf-8') as req:
        for pkg in externals:
            req.write(pkg + '\n')
    print("Generated requirements.txt with:", externals)

if __name__ == '__main__':
    main()
