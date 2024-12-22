from config import PREDICT_WINDOW_SIZE, TRAINING_SAMPLE_LOOP, TRAIN_DATA_COUNT, TRAIN_DATA_FOLDER_NAME
from ml import GestureRecognitionModel
from sample import Sampler, DataHandler, TrainDataSampler
import time
import socket
from udp_sender import send_udp_message

if __name__ == '__main__':
    sampler = Sampler()
    data_handler = DataHandler()

    # sample training data
    tds = TrainDataSampler(sampler=sampler, data_handler=data_handler)
    tds.read(loop_time=TRAINING_SAMPLE_LOOP, data_count=TRAIN_DATA_COUNT)

    print("Sample Data Done!")
    print("=" * 50)
    time.sleep(1)

    # train model
    grm = GestureRecognitionModel()
    print("Start to train model")
    grm.train()
    print("Training Done!")
    print("=" * 50)
    time.sleep(1)

    # start prediction
    print("Start to predict!")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    sampler.reset_connection()
    while True:
        # read serial data
        new_data = sampler.read_data(count=PREDICT_WINDOW_SIZE, used_for_training=False)

        # predict
        result = grm.predict(x=new_data)

        if result == "IDLE":
            send_udp_message(sock, "0", 0)

        if result == "FIST":
            send_udp_message(sock, "1", 0)
