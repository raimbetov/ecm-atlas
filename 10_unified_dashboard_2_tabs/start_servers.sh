#!/bin/bash

echo "========================================================================"
echo "ECM Atlas - Unified Dashboard v2"
echo "========================================================================"
echo ""
echo "Starting servers..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Start API server in background
echo "🚀 Starting Unified API server on port 5004..."
python3 api_server.py &
API_PID=$!
sleep 3

# Start HTTP server in background
echo "🌐 Starting HTTP server on port 8083..."
python3 -m http.server 8083 &
HTTP_PID=$!
sleep 1

echo ""
echo "========================================================================"
echo "✅ Servers started successfully!"
echo "========================================================================"
echo ""
echo "📊 Dashboard URL: http://localhost:8083/dashboard.html"
echo "🔌 API Server:    http://localhost:5004"
echo ""
echo "📝 Main Features:"
echo "   🔬 Individual Dataset Analysis - Detailed analysis of each dataset"
echo "   📊 Compare Datasets - Cross-dataset protein comparison"
echo ""
echo "📝 API Endpoints:"
echo "   - GET /api/health"
echo "   - GET /api/global_stats"
echo "   - GET /api/datasets"
echo "   - GET /api/dataset/<name>/* (individual analysis)"
echo "   - GET /api/compare/* (cross-dataset comparison)"
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
