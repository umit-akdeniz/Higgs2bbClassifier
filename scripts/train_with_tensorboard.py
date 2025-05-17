import os
from datetime import datetime
from keras.callbacks import TensorBoard

log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks.append(tensorboard_callback)
