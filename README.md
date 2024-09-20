# CRAIC2024 Aelos 自动机器人代码
相比于官方代码，主要做出了如下改进：
- 优化了自动机器人识别的阈值，快速对齐 Tag 码，缩短时间。
- 一次能搬一个或者两个方块，如果要搬两个的话要求方块靠在一次。
- 执行搬起动作后会利用头部摄像头检测方块是否成功搬起，如果失败则继续搬。
- 利用 [YOLOv8](https://docs.ultralytics.com/) 模型识别方块（主要是因为实验室的橙色方块颜色较浅，用 OpenCV 的话容易和地面混淆成一团，决赛场地的橙色方块颜色很深，没有这个烦恼）。

各文件的功能：
- `botec_code.py` 和 `image_Tag_converter.py`：部署在机器人上的代码。没法在本地运行，只是方便修改以及备份。
```bash
python botec_code.py # 正常运行，到基地区后自动摔倒
python botec_code.py -b 1 # 到基地区后回来
python botec_code.py -d 1 # 调试模式，返回机器人胸前摄像头的画面。需要运行 receiver.py 后机器人才能启动。
python botec_code.py -s 1 # 保存搬箱子前胸部摄像头的画面，用来制作数据集。也会保存执行搬箱子动作后头部摄像头看到的画面。
```
- `best.onnx`：最终训练好的模型。
- `train.ipynb`：用于训练 YOLO 模型。
- `receiver.py`：用于接收在调试模式下机器人胸部摄像头看到的图像。
- `simulation.ipynb`：用于模拟运行部分部署在机器人上的代码。
- `yolov8.yaml`：基于 `YOLOv8n` 缩小规模后的模型框架，用于训练部署在机器人上的模型。