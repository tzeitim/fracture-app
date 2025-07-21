#!/usr/bin/env python
"""
Run script for the FRACTURE Explorer Shiny application.
This script initializes and runs the Shiny app.
"""

import sys
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("fracture-app")

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
        run_app('app.py', host="0.0.0.0", port=port)
    except Exception as e:
        logger.error(f"Error launching app: {e}")
        sys.exit(1)