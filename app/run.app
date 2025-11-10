#!/usr/bin/env python
"""
Run script for the FRACTURE Explorer Shiny application.
This script initializes and runs the Shiny app.
"""

import sys
import os
from pathlib import Path
import logging
import argparse

# Configure logging to match uvicorn's style for startup messages
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(message)s"
)
logger = logging.getLogger()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='FRACTURE Explorer - Assembly graph visualization')
    parser.add_argument('--port', type=int, default=None,
                        help='Port to run the server on (default: 8000 or PORT env var)')
    parser.add_argument('--file_path', type=str, default=None,
                        help='Path to parquet file to load automatically (optional)')
    parser.add_argument('--start_anchor', type=str, default="GAGACTGCATGG",
                        help='Default sequence for Start Anchor (5\' end)')
    parser.add_argument('--end_anchor', type=str, default="TTTAGTGAGGGT",
                        help='Default sequence for End Anchor (3\' end)')
    parser.add_argument('--umi', type=str, default=None,
                        help='Default UMI to select (optional)')
    parser.add_argument('--assembly_method', type=str, default="shortest_path",
                        choices=['compression', 'shortest_path'],
                        help='Assembly method (default: shortest_path)')
    parser.add_argument('--min_coverage', type=int, default=5,
                        help='Minimum coverage threshold (default: 5)')
    parser.add_argument('--kmer_size', type=int, default=10,
                        help='K-mer size for assembly (default: 10)')
    parser.add_argument('--auto_k', action='store_true',
                        help='Enable automatic k-mer size selection')
    args = parser.parse_args()

    # Add script directory to path to ensure imports work
    app_dir = Path(__file__).parent.absolute()
    sys.path.append(str(app_dir))

    # Change to the app directory to ensure relative paths work
    os.chdir(app_dir)

    logger.info(f"Starting FRACTURE Explorer from directory: {app_dir}")

    # Import and run the app
    try:
        from shiny import run_app

        # Get port from command line argument, environment variable, or default to 8000
        if args.port is not None:
            port = args.port
        else:
            port = int(os.environ.get("PORT", 8000))

        logger.info(f"Launching app on port {port}")

        # Set configuration in environment so app.py can access it
        os.environ["FRACTURE_APP_PORT"] = str(port)
        os.environ["FRACTURE_START_ANCHOR"] = args.start_anchor
        os.environ["FRACTURE_END_ANCHOR"] = args.end_anchor
        os.environ["FRACTURE_ASSEMBLY_METHOD"] = args.assembly_method
        os.environ["FRACTURE_MIN_COVERAGE"] = str(args.min_coverage)
        os.environ["FRACTURE_KMER_SIZE"] = str(args.kmer_size)
        os.environ["FRACTURE_AUTO_K"] = str(args.auto_k)
        if args.umi:
            os.environ["FRACTURE_DEFAULT_UMI"] = args.umi
        if args.file_path:
            os.environ["FRACTURE_FILE_PATH"] = args.file_path

        logger.info(f"Using Start Anchor: {args.start_anchor}")
        logger.info(f"Using End Anchor: {args.end_anchor}")
        logger.info(f"Assembly Method: {args.assembly_method}")
        logger.info(f"Min Coverage: {args.min_coverage}")
        logger.info(f"K-mer Size: {args.kmer_size}")
        logger.info(f"Auto K-mer: {args.auto_k}")
        if args.umi:
            logger.info(f"Using Default UMI: {args.umi}")
        if args.file_path:
            logger.info(f"Using File Path: {args.file_path}")

        # Log server URLs with correct port - show immediately at startup
        import socket
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            logger.info("🚀 FRACTURE Explorer is starting up!")
            logger.info(f"📍 Hostname: {hostname}")
            logger.info(f"🌐 Local IP URL: http://{local_ip}:{port}")
            logger.info(f"🌍 Hostname URL: http://{hostname}:{port}")
            logger.info(f"🏠 Localhost URL: http://localhost:{port}")
            logger.info(f"🔗 Access from same network: http://{local_ip}:{port}")
        except Exception as e:
            logger.info("🚀 FRACTURE Explorer is starting up!")
            logger.info(f"🌐 Default URL: http://localhost:{port}")
            logger.warning(f"⚠️  Could not determine hostname/IP: {e}")

        # Import app directly and run with uvicorn for timeout control
        from app import app
        import uvicorn

        # Configure longer timeouts to prevent idle disconnections
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            timeout_keep_alive=300,  # 5 minutes keep-alive
            timeout_graceful_shutdown=30,
            ws_ping_interval=30,  # WebSocket ping every 30 seconds
            ws_ping_timeout=10    # WebSocket ping timeout
        )
    except Exception as e:
        logger.error(f"Error launching app: {e}")
        sys.exit(1)
