from enum import Enum


class ModelStrategy(Enum):
    Features = 1


class GestureType(Enum):
    IDLE = "IDLE"
    FIST = "FIST"


# Arduino configuration
ARDUINO_PORT = "/dev/cu.usbmodemF412FA6A23582"
BAUD_RATE = 460800
ADC_RESOLUTION = 16383
ARDUINO_SAMPLE_RATE = 2000
ARDUINO_SERIAL_DATA_TYPE = "num"
SERIAL_DATA_NORMALIZATION = False
USING_BUTTERWORTH_FILTER = False

# Model train configuration
TRAINING_SAMPLE_LOOP = 5
TRAIN_DATA_FOLDER_NAME = 'train_data'
VALIDATION_DATA_FOLDER_NAME = 'val_data'
TRAIN_DATA_COUNT = 10000
TRAIN_STEP = 700

# Model prediction configuration
PREDICT_WINDOW_SIZE = 800

# UDP
IP_ADDRESS = "127.0.0.1"
UDP_PORT = 8888
