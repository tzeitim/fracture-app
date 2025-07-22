#!/usr/bin/env python
"""
Run script for the FRACTURE Explorer Shiny application.
This script initializes and runs the Shiny app.
"""

import sys
import os
from pathlib import Path
import logging

# Configure logging to match uvicorn's style for startup messages
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(message)s"
)
logger = logging.getLogger()

if __name__ == "__main__":
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
        port = 8000
        if len(sys.argv) > 1:
            port = int(sys.argv[1])
        else:
            port = int(os.environ.get("PORT", 8000))
        
        logger.info(f"Launching app on port {port}")
        
        # Set the port in environment so app.py can access it
        os.environ["FRACTURE_APP_PORT"] = str(port)
        
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
        
        run_app('app.py', host="0.0.0.0", port=port)
    except Exception as e:
        logger.error(f"Error launching app: {e}")
        sys.exit(1)