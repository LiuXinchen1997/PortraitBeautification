# PortraitBeautification

## 运行环境

### python与python packet

* python3
* PyQt5: `pip install PyQt5`
* dlib: `pip install dlib`失败可尝试指定dlib版本，例如`pip install dlib==19.6.1`（已测试该版本dlib可支持本项目运行）
  * MacOS用户在安装dlib前可能需要安装CMake。可尝试`sudo brew install cmake`或`brew install cmake`。
* OpenCV: `pip install opencv-python`

### 其他文件

* 将人脸关键点数据文件[`shape_predictor_68_face_landmarks.dat`](http://dlib.net/files/)放进`./data/`中

## 运行

### 文件

* 顶层文件ui.py，运行ui.py，应可见带有GUI的窗口。
* util.py，主要后台处理代码。

### GUI操作

1. `打开文件`，应可见图片显示在界面中；
2. 滑动滑轨，点击滑轨前按钮（美白/瘦脸……），图片被美化。

## 计划

1. 美白：done，可测试
2. 红唇：done，可测试
3. 大眼：done，可测试
4. 瘦脸：done，可测试
5. 浓眉
