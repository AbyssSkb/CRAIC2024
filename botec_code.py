import time
import cv2
import threading
import numpy as np
import rospy
import argparse
import sys
import socket
import struct
from datetime import datetime

from image_Tag_converter import ImgConverter
from image_Tag_converter import TagConverter

sys.path.append("/home/lemon/catkin_ws/src/aelos_smart_ros")
from leju import base_action


#定义一些参数
Chest_img = None
Head_img = None

marker = None

box_x = None
box_y = None
box_w = None
box_h = None

forward = 0

# 不同色块的hsv范围
color_range = {
    'green': [(45, 40, 60), (51, 255, 255)],
    'orange': [(8, 120, 120), (18, 255, 255)]
}

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--back", default=0, type=int)
parser.add_argument("-d", "--debug", default=0, type=int)
parser.add_argument("-s", "--save", default=0, type=int)
args = parser.parse_args()

HOST = '192.168.3.41'
PORT = 9999

# 创建TCP套接字
if args.debug == 1:
   server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   server.connect((HOST, PORT))

# ******************************************动作函数***********************************************

def ide_Rturn01(n):
    for i in range(0, n):
        base_action.action("Rturn03")
        time.sleep(0.5)

# 放下箱子
def Box_Down(n):   
    for i in range (0, n):
        base_action.action("PutCubeDown")
        time.sleep(0.5)

def boxForward(n):
    base_action.action("forward1")
    for i in range(n):
        base_action.action("forward2")
    base_action.action("forward3")

# 前进
def moveForward(n, step, box):
    if n <= 0:
        return
    global forward
    forward += n * step * (box + 1)
    if box == 1:
        boxForward(n * step)
        time.sleep(0.5)
        return

    if step == 1:
        if box == 0:
            print(f'空手小步前进{n}步')
            for i in range (0, n):
                base_action.action("Forwalk01")
                time.sleep(0.2)
        else:
            print(f'抱箱小步前进{n}步')
            for i in range (0, n):
                base_action.action("BoxForward02v3")
                time.sleep(0)
    elif step == 2:
        if box == 0:
            print(f'空手大步前进{n}步')
            for i in range (0, n):
                base_action.action("FastForward1s")
                time.sleep(0.5)
        else:
            print(f'抱箱大步前进{n}步')
            for i in range (0, n):
                base_action.action("BoxForward2sv4")
                time.sleep(0.5)

    elif step == 3:
        print(f'抱箱超大步前进{n}步')
        for i in range (0, n):
                base_action.action("BoxForward3sv2")   
                time.sleep(0.5)

# 后退
def moveBackward(n, box):
    if box == 0:
        print(f'空手后退{n}步')
        for i in range (0, n):
            base_action.action("Back1Run")
            time.sleep(0)
    else:
        print(f'抱箱后退{n}步')
        for i in range (0, n):
                base_action.action("BoxBack1Runv4")
                time.sleep(0)

# 左移
def moveLeft(n, step, box):
    if step == 1:
        if box == 0:
            print(f'空手小步左移{n}步')
            for i in  range (0, n):
                base_action.action ("Left02move")
                time.sleep(0)
        else:
            print(f'抱箱小步左移{n}步')
            for i in  range (0, n):
                base_action.action ("BoxLeft1v3")
                time.sleep(0)
    else:
        if box == 0:
            print(f'空手大步左移{n}步')
            for i in  range (0, n):
                base_action.action('move_left')
                time.sleep(0.5)
        else:
            print(f'抱箱大步左移{n}步')
            for i in  range (0, n):
                base_action.action('Box_move_leftv3')
                time.sleep(0)

# 右移
def moveRight(n, step, box):
    if step == 1:
        if box == 0:
            print(f'空手小步右移{n}步') 
            for i in  range (0, n):
                base_action.action ("Right02move")
                time.sleep(0)
        else:
            print(f'抱箱小步右移{n}步')
            for i in  range (0, n):
                base_action.action ("BoxRight1v3")
                time.sleep(0)
    else:
        if box == 0:
            print(f'空手大步右移{n}步')
            for i in  range (0, n):
                base_action.action('move_right')
                time.sleep(0)
        else:
            print(f'抱箱大步右移{n}步')
            for i in  range (0, n):
                base_action.action('Box_move_rightv3')
                time.sleep(0)

# 左转
def turnLeft(n, step, box):
    if step == 1:
        if box == 0:
            print(f'空手小幅度左转{n}次')
            for i in  range (0, n):
                base_action.action("ide_turn003L")
                time.sleep(0)
        else:
            print(f'抱箱小幅度左转{n}次')
            for i in  range (0, n):
                base_action.action("ide_BoxTurnL1v3")
                time.sleep(0)
    else:
        if box == 0:
            print(f'空手大幅度左转{n}次')
            for i in range (0, n):
                base_action.action("LeftTurn1s")
                time.sleep(0.5)
        else:
            print(f'抱箱大幅度左转{n}次')
            for i in  range (0, n):
                base_action.action("BoxLeftTurn1sv3")
                time.sleep(0)

# 右转
def turnRight(n, step, box):
    if step == 1:
        if box == 0:       
            print(f'空手小幅度右转{n}次')
            for i in  range (0, n):
                base_action.action("ide_turn003R")
                time.sleep(0)
        else:
            print(f'抱箱小幅度右转{n}次')
            for i in  range (0, n):
                base_action.action("ide_BoxTurnR1v3")
                time.sleep(0)
    else:
        if box == 0:
            print(f'空手大幅度右转{n}次')
            for i in  range (0, n):
                base_action.action("RightTurn1s")
                time.sleep(0)
        else:
            print(f'抱箱大幅度右转{n}次')
            for i in  range (0, n):
                base_action.action("BoxRightTurn1sv3")
                time.sleep(0)

#获取图像
def get_img():
    global Chest_img, Head_img
    global ret
    image_reader = ImgConverter()
    while True:
        ret, Chest_img = image_reader.chest_image()
        ret, Head_img = image_reader.head_image()
        time.sleep(0.3)
        if Chest_img is not None:
            if args.debug == 1:
                result, imgencode = cv2.imencode('.jpg', Chest_img, [cv2.IMWRITE_JPEG_QUALITY, 50])

                # 发送图像数据长度
                data = imgencode.tobytes()
                length = struct.pack('!I', len(data))
                server.sendall(length)

                # 发送图像数据
                server.sendall(data)
        else:
            print("暂时未获取到图像")

# 启动一个线程专门运行 get_img
th2 = threading.Thread(target=get_img)
th2.setDaemon(True)
th2.start()

# 加载网络
net = cv2.dnn.readNet('best.onnx')
input_size = (320, 320)

#查找方块
def find_box(img):
    is_box_found = False
    global box_x, box_y, box_w, box_h
    box_x = box_y = box_w = box_h = None
    if img is None:
        print('等待获取图像中...')
        time.sleep(0.3)
    else:
        if args.save == 1:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'chest/image_{current_time}.png'
            cv2.imwrite(filename, img)

        start_time = time.time()
        height, width = 480, 640
        blob = cv2.dnn.blobFromImage(img, 1/255.0, input_size, swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()
        boxes = []
        confidences = []
        class_ids = []

        outputs = outputs[0].transpose(1, 0)

        for detection in outputs:
            scores = detection[4:]
            x, y, w, h = detection[:4]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > 0.5:
                # 计算边界框
                x = int((x - w / 2) / input_size[0] * width)
                y = int((y - h / 2) / input_size[1] * height)
                w = int(w / input_size[0] * width)
                h = int(h / input_size[1] * height)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(classId)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.6)
        if len(indices) == 0:
            return False
        else:
            is_box_found = True

        boxes = [boxes[i] for i in indices]
        dis = [(i, (x + w // 2 - 320) ** 2 + (480 - y - h) ** 2) for i, (x, y, w, h) in enumerate(boxes)]
        dis_sorted = sorted(dis, key=lambda x: x[1])
        x, y, box_w, box_h = boxes[dis_sorted[0][0]]
        box_x = x + box_w // 2
        box_y = y + box_h // 2
        end_time = time.time()
        print(f'识别成功，花费 {(end_time - start_time): .2f} 秒')
        print(f'Box: x={box_x} y={box_y} w={box_w} h={box_h}')

    return is_box_found

#搬箱子
def goto_box():
    if box_x is None :
        print('等待中获取坐标中...')
        time.sleep(0.3)
    else:
        # 左右大步移动
        if box_x < 200:
            moveLeft(1, step=2, box=0)
        elif box_x > 420:
            moveRight(1, step=2, box=0)

        # 前后移动
        elif box_y < 310: 
            moveForward(1, step=2, box=0)
        # 方块叠起来的情景
        elif box_y < 340 and box_w > 200 and box_h > 200:
            moveForward(1, step=1, box=0)
        # 一个方块的情景
        elif box_y < 380 and not (box_w > 200 and box_h > 200):
            moveForward(1, step=1, box=0)
        elif box_y >= 440:
            moveBackward(1, box=0)

        # 左右小步移动
        elif box_x < 240:
            moveLeft(1, step=1, box=0)
        elif box_x > 400:
            moveRight(1, step=1, box=0)
        
        # 对齐完毕
        else:
            print("尝试抱箱子")
            base_action.action("抱起方块v6")
            # base_action.action("抱起方块v22")
            return True
    return False

def check_box(img):
    if args.save == 1:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'head/image_{current_time}.png'
        cv2.imwrite(filename, img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(img, color_range['green'][0], color_range['green'][1])
    mask2 = cv2.inRange(img, color_range['orange'][0], color_range['orange'][1])
    mask = cv2.bitwise_or(mask1, mask2)
    box_img_opened = cv2.dilate(mask, np.ones((4, 4), np.uint8), iterations=2)
    mask = box_img_opened[430:480, 240:440]
    radio = cv2.countNonZero(mask) / mask.size
    print(f"radio: {radio}")
    return radio > 0.5


# ***************************************tag对正****************************************
def turn_to_tag(dis_x, dis_y, theta, x_offset=0.15, y_offset=0.0, theta_offset=0.0, x_threshold=0.04, y_threshold=0.03, theta_threshold=10.0):
    is_turn_done = False
    box = 2 - direction
    x_error = dis_x - x_offset
    y_error = dis_y - y_offset
    theta_error = theta - theta_offset
    if theta_error > 180:
        theta_error -= 360

    print("ID:", marker[0], "x_error:", x_error, "y_error:", y_error, "theta_error:", theta_error)

    # 后退
    if (x_error < -x_threshold):
        print("x_error: ", x_error, " < ", -x_threshold)
        moveBackward(1, box=box)
    
    # 大幅度左右转
    elif (theta_error > max(theta_threshold, 32 - theta_threshold)):
        print("theta_error: ", theta_error, " > ", max(theta_threshold, 32 - theta_threshold))
        turnLeft(1, step=2, box=box)
    elif (theta_error < min(-theta_threshold, -32 + theta_threshold)):
        print("theta_error: ", theta_error, " < ", min(-theta_threshold, -32 + theta_threshold))
        turnRight(1, step=2, box=box)

    # 大步左右移动
    elif (y_error > y_threshold + 0.02):
        print("y_error: ", y_error, " > ", y_threshold + 0.02)
        moveLeft(1, step=2, box=box)
    elif (y_error < -y_threshold - 0.02):
        print("y_error: ", y_error," < ", -y_threshold - 0.02)
        moveRight(1, step=2, box=box)

    # 前进
    elif (x_error > max(x_threshold, 0.16 - x_threshold)):
        print("x_error: ", x_error, "> ", max(x_threshold, 0.16 - x_threshold))
        moveForward(1, step=2, box=box)
    elif (x_error > x_threshold):
        print("x_error: ", x_error, " > ", x_threshold)
        moveForward(1, step=1, box=box)
    
    # 小幅度左右转
    elif (theta_error > theta_threshold):
        print("theta_error: ", theta_error, " > ", theta_threshold)
        turnLeft(1, step=1, box=box)
    elif (theta_error < -theta_threshold):
        print("theta_error: ", theta_error, " < ", -theta_threshold)
        turnRight(1, step=1, box=box)
    
    # 小步左右移动
    elif (y_error > y_threshold):
        print("y_error: ", y_error, " > ", y_threshold)
        moveLeft(1, step=1, box=box)
    elif (y_error < -y_threshold):
        print("y_error: ", y_error," < ", -y_threshold)
        moveRight(1, step=1, box=box)

    # 对准完毕
    else:    
        is_turn_done = True 

    return is_turn_done


if __name__ == '__main__':
    start_time = time.time()
    rospy.init_node('image_listener')   #ROS节点初始化
    Tag = TagConverter()
    time.sleep(1)
    ID = 0
    direction = 1
    
    level = "start_box"
    # time.sleep(5)
    while Chest_img is None:
        print('waiting...')
        time.sleep(0.1)

    print('Back:', args.back)
    print('Debug:', args.debug)
    print('Save: ', args.save)
    print('原神，启动！')
    moveForward(3, step=2, box=0)
    # moveLeft(2, step=2, box=0)
    # Debug
    notag = 0
    while not rospy.is_shutdown():
        now_time = time.time()
        print(f'时刻：{now_time - start_time}')
        if ID == 0:  #搬箱子
            if level == "start_box":
                time.sleep(0.3)
                if find_box(Chest_img.copy()):
                    if goto_box():
                        time.sleep(0.3)
                        if check_box(Head_img.copy()):
                            print('成功抱起箱子')
                            level = "end_box"
                else:
                    moveForward(1, step=2, box=0)
                    # moveLeft(1, step=2, box=0)
            elif level == "end_box":
                if direction == 1:
                    # 竖着走
                    a = (24 - forward) // 6
                    b = 24 - forward - 4 * a
                    moveForward(a, step=3, box=1)
                    moveForward(b // 4, step=2, box=1)
                    turnRight(4, step=2, box=1)
                    # 横着走
                    # moveLeft(4, step=2, box=1)
                    # moveForward(2, step=2, box=1)
                elif direction == 2:
                    moveForward(1, step=1, box=1)
                    turnLeft(10, step=2, box=1)
                    direction = 1
                ID += 1
        else:     
            marker = Tag.get_nearest_marker()
            if len(marker) == 0:
                print("无tag")
                notag += 1
                if notag == 1:
                    time.sleep(0.5)
                    continue
                notag = 0
                if (ID == 1 and level == "end_box"):
                    moveForward(1, step=2, box=1)
                elif ID == 1 and level == "start_moving":
                    moveForward(1, step=2, box=1)
                elif ID == 2 and direction == 1:
                    moveForward(1, step=2, box=1)
                elif ID == 3:
                    moveBackward(1, box=1)
                elif ID == 4:
                    moveForward(1, step=2, box=1)
                elif (ID == 5 and direction == 2 and level == "start_moving"):
                    turnRight(1, step=2, box=0)
                elif (ID == 5 and direction == 2 and level == "reverse_moving"):
                    moveBackward(1, box=0)
                elif ID == 7:
                    moveForward(1, step=2, box=0)
                elif ID == 1 and direction == 2:
                    moveRight(1, step=2, box=0)
                elif ID == 6:
                    moveForward(1, step=2, box=0)
                elif ID == 5 and direction == 1:
                    # pass
                    moveLeft(1, step=2, box=1)
                              
            else:
                notag = 0
                robot_tag_x = marker[1]
                robot_tag_y = marker[2]
                tag_yaw = marker[3] + 90 # artag 正方向与机器人正方向对齐
                print(f'识别到了{marker[0]}号码')
                if direction == 1:
                    if marker[0] == 1:
                        if ID == 1:
                            level = "start_moving"
                            result = turn_to_tag(robot_tag_x, robot_tag_y, tag_yaw, x_threshold=0.08, y_threshold=0.04, x_offset=0.10)
                            if result == True:
                                print('一号码对正完毕，前进对正二号码')
                                ID += 1
                                # moveForward(3, step=2, box=1)
                                moveForward(2, step=3, box=1)
                        else:
                            moveForward(2, step=2, box=1)

                    elif marker[0] == 2:
                        if ID == 2:
                            result = turn_to_tag(robot_tag_x, robot_tag_y, tag_yaw, y_threshold=0.08, x_threshold=0.06, x_offset=0.16, y_offset=0.04)
                            if result == True:
                                print('二号码对正完毕，右移对正三号码')
                                ID += 1
                                moveRight(4, step=2, box=1)
                        elif ID == 3:
                            moveRight(1, step=2, box=1)
                        elif ID == 1:
                            ID = 2

                    elif marker[0] == 3:
                        if ID == 3:
                            result = turn_to_tag(robot_tag_x, robot_tag_y, tag_yaw, y_threshold=0.06, x_threshold=0.06, y_offset=0.02, x_offset=0.18)
                            if result == True:
                                print('三号码对正完毕，右移后前进对正四号码')
                                ID += 1
                                moveRight(3, step=2, box=1)
                                moveForward(3, step=3, box=1)
                        elif ID == 4:
                            moveRight(1, step=2, box=1)
                        elif ID == 7:
                            turnLeft(1, step=2, box=1)
                            
                    elif marker[0] == 4:
                        if ID == 4:
                            result = turn_to_tag(robot_tag_x, robot_tag_y, tag_yaw, y_threshold=0.1, x_threshold=0.03, x_offset=0.21)
                            if result == True:
                                print('四号码对正完毕，左移对正五号码')
                                ID += 1
                                moveLeft(5, step=2, box=1)
                        elif ID == 5:
                            moveLeft(1, step=2, box=1)

                    elif marker[0] == 5:
                        if ID == 5:
                            # continue
                            result = turn_to_tag(robot_tag_x, robot_tag_y, tag_yaw, x_threshold=0.1, y_threshold=0.06, theta_threshold=16.0)
                            if result == True:
                                print('五号码对正完毕，前进至大本营并放下海绵块')
                                moveForward(2, step=3, box=1)
                                direction = 2
                                if args.back == 0:
                                    base_action.action("falldown4v2")
                                    break
                                ide_Rturn01(1)
                        elif ID == 4:
                            ID = 5
            
                elif direction == 2:
                    #反方向
                    if marker[0] == 4:
                        if ID == 5:
                            turnRight(1, step=2, box=0)

                    elif marker[0] == 5:
                        result = turn_to_tag(robot_tag_x, robot_tag_y, tag_yaw, theta_offset=180.0, x_threshold=0.06, theta_threshold=4.0) # 5号为反方向
                        level = "reverse_moving"
                        if result == True:
                            print('五号码反向对正完毕，前进对正六号码')
                            ID += 1
                            moveForward(2, step=2, box=0)
                            moveForward(1, step=1, box=0)

                    elif marker[0] == 6:
                        if ID == 6:
                            result = turn_to_tag(robot_tag_x, robot_tag_y, tag_yaw, x_threshold=0.06, theta_threshold=4.0, y_offset=0.02)
                            if result == True:
                                print('六号码对正完毕，左移对正七号码')
                                ID += 1
                                moveLeft(7, step=2, box=0)
                                turnRight(1, step=1, box=0)
                                moveForward(7, step=2, box=0)
                        elif ID == 5:
                            ID += 1
                    
                    elif marker[0] == 7:
                        if ID == 7:
                            result = turn_to_tag(robot_tag_x, robot_tag_y, tag_yaw, theta_threshold=4.0, x_threshold=0.06)
                            if result == True:
                                print('七号码对正完毕，右移反向对正一号码')
                                moveRight(6, step=2, box=0)
                                ID = 1
                                #turnRight(2,step=2,box=0)
                                #moveRight(6, step=2, box=0)

                    elif marker[0] == 1:
                        result = turn_to_tag(robot_tag_x, robot_tag_y, tag_yaw, theta_offset=180, x_threshold=0.06) # 1号为反方向
                        if result == True:
                            print('一号码反向对正完毕，前进抓取海绵块')
                            moveForward(2, step=2, box=0)
                            ID = 0
                            level = "start_box"
        print()
        time.sleep(0.1)
    end_time = time.time()
    print(f"运行时间：{(end_time - start_time): .2f} 秒")