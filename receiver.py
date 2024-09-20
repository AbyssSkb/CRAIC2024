import cv2
import numpy as np
import socket
import struct
from ultralytics import YOLO
from threading import Thread
import queue

HOST = ''
PORT = 9999


# 创建TCP服务器套接字
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print('Now waiting for connections...')
conn, addr = server.accept()
print(f'Connected by {addr}')

model = YOLO("runs/detect/train18/weights/best.pt")

# 创建队列用于存储接收到的图像数据
data_queue = queue.Queue()

def receive_data():
    try:
        while True:
            # 接收图像数据长度
            data_length = conn.recv(4)
            if not data_length:
                break
            length = struct.unpack('!I', data_length)[0]

            # 接收图像数据
            data = b''
            while len(data) < length:
                packet = conn.recv(length - len(data))
                if not packet:
                    break
                data += packet
            
            # 将接收到的数据放入队列
            data_queue.put(data)
    finally:
        conn.close()

def process_data():
    try:
        while True:
            # 从队列中获取图像数据
            data = data_queue.get()
            if data is None:
                break

            # 解码图像
            data = np.frombuffer(data, dtype=np.uint8)
            imgdecode = cv2.imdecode(data, cv2.IMREAD_COLOR)

            # YOLO目标检测
            results = model(imgdecode, imgsz=320, conf=0.5, iou=0.6)

            for result in results:
                img = result.plot()
                cv2.imshow("frames", img)

            if cv2.waitKey(1) == 27:  # 按下“ESC”退出
                break
    finally:
        cv2.destroyAllWindows()

# 创建并启动线程
receiver_thread = Thread(target=receive_data)
processor_thread = Thread(target=process_data)

receiver_thread.start()
processor_thread.start()

# 等待线程结束
receiver_thread.join()
processor_thread.join()

# 清理资源
server.close()
