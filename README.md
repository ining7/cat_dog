[Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview)

本代码运行于windows环境

## 数据预处理

- 数据来源：https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data?select=train.zip、
- 在`train/config.py`中修改`base_dier`

```shell
cd train
python init_data.py
```

## 训练

​	在`train/config.py`中修改训练基本信息

```shell
cd train
python training.py
```

## 过程与结果数据

- 模型结果保存在`./check_point`
- 训练过程中`loss`与`acc`图保存于`./image`下对应文件夹中

## 推理


### Python
```shell
cd train
# inference
python inference.py --model_path ../check_point/cat_dog.pth --image_path ../data/test/cat.2.jpg
# convert
python inference.py --model_path ../check_point/cat_dog.pth --image_path ../data/test/cat.2.jpg --convrt_path ../onnx_model
# check_acc
python inference.py --model_path ../check_point/cat_dog.pth --image_path ../data/test/cat.2.jpg --onnx_path ../onnx_model
```

### C++
```shell
cd inference
# convert .pt to .bin
python pt2bin.py --input ../train/pt_file/cat_dog_input.pt --output ./torch_cat_dog_input.bin
# run inference in C++
mkdir build && cd build
cmake ..
make
./cat_dog ../../onnx_model/cat_dog.onnx ../torch_cat_dog_input.bin ../onnx_cat_dog_output.bin
# check accuracy
cd ..
python check_acc_bin.py --pt_file ../train/pt_file/cat_dog_output.pt --bin_file ./onnx_cat_dog_output.bin
```
