# Unconditional Diffusion Model

本仓库实现了比较简单的无条件扩散模型,。
Diffusion类中的训练与测试方法按照Denoising Diffusion Probabilistic Models论文实现。UNet架构参考了Stable Diffusion中的实现并进行了简化：

![Alt text](<structure.jpg>)

## Setup

你可以运行如下代码来安装所需依赖：

```shell
pip install -r requirements.txt
```

## Repository structure

```plain
|-- data/ # 存放所需数据
    |-- archive/ # ImageFolder需要子文件夹
|-- diffusion # Diffusion类定义
|-- main.py # 模型训练与推理
|-- module.py # UNet所需组件
|-- requiremens.txt # 需要的依赖
|-- test.ipynb # 复现时的一些测试代码
|-- train.py # 定义模型训练函数
|-- UNet.py # UNet定义
|-- utils.py # 辅助工具
```

## Run pipeline

```plain
Optional arguments:
  --run_name RUN_NAME   Name of the run (default: DDPM_Unconditional)
  --epochs EPOCHS       Number of epochs (default: 1)
  --batch_size BATCH_SIZE
                        Batch size (default: 8)
  --image_size IMAGE_SIZE
                        Image size (default: 64)
  --dataset_path DATASET_PATH
                        Path to the dataset (default: ./data)
  --device DEVICE       Device for training (default: cuda)
  --lr LR               Learning rate (default: 0.0003)
```

如果你想在自己的数据集上运行的话，请将dataset_path指定为你的数据集，你的数据集需要遵守ImageFolder的要求。

示例：

```shell
python main.py --dataset_path "./data"
```

## Attribution

<https://github.com/dome272/Diffusion-Models-pytorch>
<https://zhuanlan.zhihu.com/p/642354007>
<https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html>
