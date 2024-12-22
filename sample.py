import os
import time
import serial
import pandas as pd
from config import TRAIN_DATA_FOLDER_NAME, ARDUINO_PORT, BAUD_RATE, GestureType
import struct

COUNTDOWN = 3


class Sampler:
    def __init__(self):
        self.ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)

    def read_data(self, count: int, used_for_training: bool = False):
        # ⚠️ important!
        if used_for_training:
            self.reset_connection()

        result = []
        cnt = 0

        while cnt < count:
            result.append(list(self.__read_one_record()))
            cnt += 1

        return result

    def __read_one_record(self):
        while True:
            # read sync header
            header1 = self.ser.read(1)
            if header1 != b'\x7F':
                continue

            header2 = self.ser.read(1)
            if header2 == b'\xFF':
                break

        # read analog data
        data = self.ser.read(6)
        if len(data) != 6:
            raise Exception("reading analog data wrong!")

        ch1_val = struct.unpack('>H', data[0:2])[0]
        ch2_val = struct.unpack('>H', data[2:4])[0]
        ch3_val = struct.unpack('>H', data[4:6])[0]

        return ch1_val, ch2_val, ch3_val

    def reset_connection(self):
        self.ser.close()
        time.sleep(1)
        self.ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)


class DataHandler:
    def __init__(self):
        pass

    @staticmethod
    def save_to_csv_file(data: list, dirname: str, filename: str):
        target_path = os.path.join(dirname, filename)
        pd.DataFrame(data).to_csv(target_path, index=False, header=False)

    @staticmethod
    def clear_folder(dirname: str):
        for filename in os.listdir(dirname):
            target_path = os.path.join(dirname, filename)
            os.remove(target_path)


class TrainDataSampler:
    def __init__(self, sampler: Sampler, data_handler: DataHandler):
        self.sampler = sampler
        self.data_handler = data_handler

    def read(self, loop_time: int, data_count: int):
        # delete existed data file
        self.data_handler.clear_folder(TRAIN_DATA_FOLDER_NAME)

        # sample
        cnt = 0
        while cnt < loop_time:
            self.__sample_specific_gesture_data(gesture=GestureType.IDLE, count=data_count)
            self.__sample_specific_gesture_data(gesture=GestureType.FIST, count=data_count)
            cnt += 1

    def __sample_specific_gesture_data(self, gesture: GestureType, count: int):
        # 1. show notification
        self.__show_sampling_notification(gesture)

        # 2. read data
        data = self.sampler.read_data(count=count, used_for_training=True)

        # 3. save data
        filename = f'{gesture.value}_{int(time.time())}.csv'
        self.data_handler.save_to_csv_file(data=data, dirname=TRAIN_DATA_FOLDER_NAME, filename=filename)

    @staticmethod
    def __show_sampling_notification(gesture: GestureType):
        print('ready to read "%s"' % gesture.value)
        time.sleep(1)

        for i in range(COUNTDOWN, 0, -1):
            print(f"\r{i}", end="")
            time.sleep(1)

        print("\rstart sampling!")
