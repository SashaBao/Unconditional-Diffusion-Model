# Unconditional Diffusion Model

���ֿ�ʵ���˱Ƚϼ򵥵���������ɢģ��,��
Diffusion���е�ѵ������Է�������Denoising Diffusion Probabilistic Models����ʵ�֡�UNet�ܹ��ο���Stable Diffusion�е�ʵ�ֲ������˼򻯣�

![Alt text](<structure.jpg>)

## Setup

������������´�������װ����������

```shell
pip install -r requirements.txt
```

## Repository structure

```plain
|-- data/ # �����������
    |-- archive/ # ImageFolder��Ҫ���ļ���
|-- diffusion # Diffusion�ඨ��
|-- main.py # ģ��ѵ��������
|-- module.py # UNet�������
|-- requiremens.txt # ��Ҫ������
|-- test.ipynb # ����ʱ��һЩ���Դ���
|-- train.py # ����ģ��ѵ������
|-- UNet.py # UNet����
|-- utils.py # ��������
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

����������Լ������ݼ������еĻ����뽫dataset_pathָ��Ϊ������ݼ���������ݼ���Ҫ����ImageFolder��Ҫ��

ʾ����

```shell
python main.py --dataset_path "./data"
```

## Attribution

<https://github.com/dome272/Diffusion-Models-pytorch>
<https://zhuanlan.zhihu.com/p/642354007>
<https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html>
