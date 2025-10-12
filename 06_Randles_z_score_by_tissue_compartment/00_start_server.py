#!/usr/bin/env python3
"""
Simple HTTP Server for Z-Score Dashboard
Serves the HTML dashboard on localhost:8080
"""

import http.server
import socketserver
import os

PORT = 8080

# Change to dashboard directory
os.chdir('/Users/Kravtsovd/projects/ecm-atlas/06_Randles_z_score_by_tissue_compartment')

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"🚀 Server started at http://localhost:{PORT}")
    print(f"📊 Dashboard URL: http://localhost:{PORT}/zscore_dashboard.html")
    print("\nPress Ctrl+C to stop server...")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        httpd.shutdown()
