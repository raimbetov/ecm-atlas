#!/bin/bash

echo "========================================================================"
echo "ECM Atlas - Unified Multi-Tissue Dashboard"
echo "========================================================================"
echo ""
echo "Starting servers..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Start API server in background
echo "🚀 Starting API server on port 5002..."
python3 api_server.py &
API_PID=$!
sleep 2

# Start HTTP server in background
echo "🌐 Starting HTTP server on port 8081..."
python3 -m http.server 8081 &
HTTP_PID=$!
sleep 1

echo ""
echo "========================================================================"
echo "✅ Servers started successfully!"
echo "========================================================================"
echo ""
echo "📊 Dashboard URL: http://localhost:8081/dashboard.html"
echo "🔌 API Server:    http://localhost:5002"
echo ""
echo "📝 API Endpoints:"
echo "   - GET /api/health"
echo "   - GET /api/stats"
echo "   - GET /api/filters"
echo "   - GET /api/heatmap?[filters]"
echo "   - GET /api/protein/<protein_id>"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "========================================================================"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $API_PID 2>/dev/null
    kill $HTTP_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup INT TERM

# Wait for processes
wait
