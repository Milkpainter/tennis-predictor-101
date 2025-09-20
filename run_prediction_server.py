#!/usr/bin/env python3
"""Run Tennis Predictor 101 API Server.

Starts the FastAPI server for making tennis predictions via REST API.
Provides real-time prediction capabilities with comprehensive analysis.
"""

import os
import sys
import argparse
import uvicorn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from api.prediction_server import app


def main():
    """Run the prediction API server."""
    
    parser = argparse.ArgumentParser(description='Tennis Predictor 101 API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--log-level', default='info', 
                       choices=['debug', 'info', 'warning', 'error'],
                       help='Log level')
    
    args = parser.parse_args()
    
    print("\n🎾 TENNIS PREDICTOR 101 API SERVER 🎾")
    print("====================================")
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Log level: {args.log_level}")
    print(f"Auto-reload: {args.reload}")
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    print("====================================")
    
    # Run the server
    uvicorn.run(
        "api.prediction_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True
    )


if __name__ == '__main__':
    main()