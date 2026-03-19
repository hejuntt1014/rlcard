#!/bin/bash
# A3 Dizhu C++ Engine 编译脚本
# 在训练服务器上运行
#
# 用法:
#   cd /root/a3dizhu_v6   (或你的 rl 目录)
#   bash build_cpp.sh
#
# 依赖:
#   - Python 3.8+ with development headers
#   - g++ 7+ or clang++ 5+
#   - pip install pybind11

set -e

echo "=== A3 Dizhu C++ Engine Build ==="

# 1. Install pybind11 if not present
python3 -c "import pybind11" 2>/dev/null || {
    echo "Installing pybind11..."
    pip3 install pybind11
}

# 2. Build
echo "Building C++ engine..."
cd "$(dirname "$0")"
python3 setup_cpp.py build_ext --inplace

# 3. Verify
echo ""
echo "Verifying..."
python3 -c "
from a3dizhu_cpp import CppEngine
e = CppEngine()
e.seed(42)
e.reset()
print('  CppEngine: OK')
obs = e.encode_obs(0)
print(f'  encode_obs shape: {obs.shape}')
legal = e.get_legal_actions()
print(f'  legal_actions: {len(legal)} actions')
print('BUILD SUCCESS')
"

echo ""
echo "=== Done! ==="
echo "The C++ module is ready. Training will auto-detect and use it."
echo "Run 'python3 test_cpp.py' for full validation."
