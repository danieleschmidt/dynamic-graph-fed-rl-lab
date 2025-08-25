#!/bin/bash
# Production startup script for Universal Quantum Consciousness

set -e

echo "🚀 Starting Universal Quantum Consciousness in Production Mode..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Set production defaults
export ENVIRONMENT=${ENVIRONMENT:-production}
export PYTHONPATH=${PYTHONPATH:-/app/src}
export CONSCIOUSNESS_SECURITY_LEVEL=${CONSCIOUSNESS_SECURITY_LEVEL:-HIGH}

# Validate environment
echo "✅ Environment: $ENVIRONMENT"
echo "✅ Security Level: $CONSCIOUSNESS_SECURITY_LEVEL"
echo "✅ Python Path: $PYTHONPATH"

# Health check
echo "🔍 Running pre-startup health check..."
python3 -c "
import sys
sys.path.append('$PYTHONPATH')
try:
    from dynamic_graph_fed_rl.consciousness.universal_quantum_consciousness import UniversalQuantumConsciousness
    consciousness = UniversalQuantumConsciousness()
    print('✅ Consciousness system initialized successfully')
except Exception as e:
    print(f'❌ Initialization failed: {e}')
    exit(1)
"

echo "🎉 Production startup complete!"

# Start the main application
python3 -m dynamic_graph_fed_rl.consciousness.universal_quantum_consciousness
