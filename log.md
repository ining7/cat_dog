### 2023-07-09

1. 更换数据集，新数据集使用init_archive.py初始化
2. image_size在config中修改
3. 修改模型layer中的参数以适应image_size，添加AdaptiveAvgPool2d

#### Next goal：加bilinear