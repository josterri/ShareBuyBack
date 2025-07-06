import subprocess
import os

base_path = os.path.dirname(__file__)
test_path = os.path.join(base_path, "tests")

print("🧪 Running all tests in:", test_path)
result = subprocess.run(["pytest", test_path], capture_output=True, text=True)

print("✅ Test output:")
print(result.stdout)

if result.returncode != 0:
    print("❌ Some tests failed.")
    print(result.stderr)
else:
    print("✅ All tests passed.")
