# Placeholder for utility functions
# e.g., logging, custom error handlers, date parsing helpers

import logging

def setup_logger():
    logger = logging.getLogger("finops_forecast_app")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = setup_logger()

# More utility functions will be added here
