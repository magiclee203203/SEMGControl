import time
from config import IP_ADDRESS, UDP_PORT


def send_udp_message(udp_sock, msg: str, interval: int):
    udp_sock.sendto(msg.encode('utf-8'), (IP_ADDRESS, UDP_PORT))
    time.sleep(interval)
