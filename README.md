本代码运行于windows环境

### 数据预处理

```shell
cd train
python init_data.py
```

### 训练

​	在`train/config.py`中修改训练基本信息

```shell
cd train
python training.py
```

### 过程与结果数据

- 模型结果保存在`./check_point`
- 训练过程中`loss`与`acc`图保存于`./image`下对应文件夹中

