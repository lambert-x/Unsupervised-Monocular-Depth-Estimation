# #####################################
# ########     Deep Learning Project      ###########
# #####################################

## Prerequisite

```bash
pip3 install -r requirements.txt
```

or install manually the following packages :

```
pytorch >= 1.0.1
pebble
matplotlib
imageio
scipy
argparse
tensorboardX
blessings
progressbar2
path.py
```

## Training
#### 从零开始训练

```bash
python3 train.py ./path/to/the/formatted/data/ -b4 -m0.2 -s0.1 --lr 2e-4 --epoch-size 3000 --sequence-length 3 --log-output [--with-gt]
```

#### pretrain (注意不同电脑的路径读取不同，可能需要根据自己的电脑系统修改(比如把'./' 变成'/'))

[Avalaible here](https://drive.google.com/drive/folders/1H1AFqSS8wr_YzwG2xWwAQHTfXN5Moxmx)
可以使用任何模型，这里就是读取了训练好的模型，

```bash
python3 train.py ./path/to/the/formatted/data/ -b4 -m0 -s2.0 --lr 2e-4 --pretrained-disp ./path/to/the/pretrain/model.pth --epoch-size 1000 --sequence-length 5 --log-output --with-gt
```

## Visualization

#### loss：
```bash
tensorboard --logdir=checkpoints/
```

#### picture：
```bash
python3 run_inference.py --pretrained ./path/to/dispnet.pth --dataset-dir ./path/pictures/dir --output-dir ./path/to/output/dir  --output-depth True
```
第一个路径是模型的路径，第二个路径是需要测试的图片的路径，第三个路径是要保存的位置, 输出深度图
