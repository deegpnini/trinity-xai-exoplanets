#!/data/data/com.termux/files/usr/bin/bash

echo "🧪 D7D_CORE QUICK TEST"
echo "======================"

# Testar imports básicos
echo "1. Testing Python imports..."
python3 -c "
try:
    import math, json, time, numpy as np
    from datetime import datetime
    print('✅ Basic imports: OK')
    
    # Testar D7D modules
    from modules import d7d_core
    print('✅ D7D modules: OK')
    
    # Testar instanciação
    config = d7d_core.D7DConfig()
    print(f'✅ D7D Config: {config.version}')
    
    # Testar numpy
    arr = np.array([1,2,3,4,5])
    print(f'✅ NumPy test: mean={arr.mean():.2f}')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "2. Testing configuration..."
if [ -f "config/d7d_config.json" ]; then
    echo "✅ Config file found"
    python3 -c "import json; data=json.load(open('config/d7d_config.json')); print(f'   Version: {data[\"d7d_core\"][\"version\"]}')"
else
    echo "⚠ Config file missing"
fi

echo ""
echo "3. Testing directory structure..."
ls -la
echo ""
echo "📁 Modules:"
ls modules/ 2>/dev/null || echo "  (creating...)"
mkdir -p modules 2>/dev/null

echo ""
echo "🎯 READY FOR D7D_CORE SIMULATION!"
echo "   Run: python d7d_main.py"
