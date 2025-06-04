# 基于 Google Colab 的图像分类模型训练与导出

## 项目概述
本项目借助 Google Colab 完成花卉图像分类模型的训练与导出，通过自定义 Python 3.9 虚拟环境解决依赖冲突。

## 环境搭建
### 安装 Python 3.9 及工具
```bash
!sudo apt-get update -y
!sudo apt-get install python3.9 python3.9-venv python3.9-distutils curl -y
```

### 创建虚拟环境并安装 pip
```bash
!python3.9 -m venv /content/tflite_env
!curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
!/content/tflite_env/bin/python get-pip.py
```

## 依赖安装
```bash
! /content/tflite_env/bin/pip install -q \
  tensorflow==2.10.0 keras==2.10.0 numpy==1.23.5 \
  protobuf==3.19.6 tensorflow-hub==0.12.0 tflite-support==0.4.2 \
  tensorflow-datasets==4.8.3 sentencepiece==0.1.99 sounddevice==0.4.5 \
  librosa==0.8.1 flatbuffers==23.5.26 matplotlib==3.5.3 \
  opencv-python==4.8.0.76 tflite-model-maker==0.4.2 \
  matplotlib_inline IPython
```

## 模型训练
```python
with open('/content/step_train.py', 'w') as f:
    f.write("""
import tensorflow as tf
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

image_path = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)
model = image_classifier.create(train_data)
loss, acc = model.evaluate(test_data)
print(f'✅ 测试准确率: {acc:.4f}')
model.export(export_dir='.')
""")
! /content/tflite_env/bin/python /content/step_train.py
```

## 模型下载
```python
from google.colab import files
files.download('model.tflite')
```

## 总结
本项目解决了依赖冲突问题，加深了对 Python 虚拟环境和库版本配合的理解，熟悉了 Colab 操作和 TFLite Model Maker 工作机制。 
