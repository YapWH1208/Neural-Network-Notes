Logger is commonly used when we are training our model on cloud, to save money while saving data of model training.

Log can save the information that we want to need, normally the performance of the model during the training process.

Code for logging function:
```python
import os
import logging

def set_logger(log_path):
	# remove the log with same name
    if os.path.exists(log_path) is True:
        os.remove(log_path)
    # Initialize log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
```

Demo usage:
```python
log_path = "./demo.log"
set_logger(log_path)
logging.info("======= Start =======")
for i in range(10):
	logging.info(f"i = {i}")
logging.info("=======  End  =======")
```

Output:
- Terminal
```terminal
======= Start =======
i = 0
i = 1
i = 2
i = 3
i = 4
i = 5
i = 6
i = 7
i = 8
i = 9
=======  End  =======
```

- log
```txt
2023-12-14 09:24:49,579:INFO: ======= Start =======
2023-12-14 09:24:49,579:INFO: i = 0
2023-12-14 09:24:49,580:INFO: i = 1
2023-12-14 09:24:49,580:INFO: i = 2
2023-12-14 09:24:49,580:INFO: i = 3
2023-12-14 09:24:49,580:INFO: i = 4
2023-12-14 09:24:49,580:INFO: i = 5
2023-12-14 09:24:49,580:INFO: i = 6
2023-12-14 09:24:49,580:INFO: i = 7
2023-12-14 09:24:49,581:INFO: i = 8
2023-12-14 09:24:49,581:INFO: i = 9
2023-12-14 09:24:49,581:INFO: =======  End  =======

```
