import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

#Build and create the logs and store them inot a folder
logs_path=os.path.join(os.getcwd(),"logs")

os.mkdir(logs_path,exists_ok=True)

LOG_FILE_PATH=os.path.join(os.getcwd(),LOG_FILE)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(lineno)d - %(levelname)s - %(message)s",
)

logging.info("Logging has started")