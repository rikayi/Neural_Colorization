# DRU_Colorization
Implimentation of paper ["Grayscale Image Colorization using deep CNN and Inception-ResNet-v2"](https://arxiv.org/abs/1712.03400) using DRU [template](https://github.com/dataroot/DRU-DL-Project-Structure/blob/master/README.md). 
## Examples
![alt text](https://github.com/rikayi/DRU_Colorization/blob/master/resources/img_38epoch65.png)
![alt text](https://github.com/rikayi/DRU_Colorization/blob/master/resources/img_48epoch65.png)
![alt text](https://github.com/rikayi/DRU_Colorization/blob/master/resources/img_49epoch65.png)
![alt text](https://github.com/rikayi/DRU_Colorization/blob/master/resources/img_4epoch65.png)
![alt text](https://github.com/rikayi/DRU_Colorization/blob/master/resources/img_7epoch65.png)
![alt text](https://github.com/rikayi/DRU_Colorization/blob/master/resources/img_45epoch65.png)

## Try it out
Clone repository
```bash
git clone https://github.com/rikayi/DRU_Colorization.git
cd DRU_Colorization
```
Install requirements
```bash
pip install -r requirements.txt
```
Download your dataset to "data/Dataset" folder and [weights](https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5) for Inception network to "data" folder.

"data" folder structure
--------------
```
└── data
     ├── Dataset
         ├── Train
         └── Test
     ├── embeds.py
     ├── reshape.py
     └── inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5

```
Reshape your dataset
```bash
python reshape.py
```
Create Inception embedings
```bash
python embeds.py
```

Change config file to your needs and start training
```bash
python example_color.py -c ../configs/color.json
```
Open one more terminal window and run tensorboard to track training process
```bash
tensorboard --logdir experiments/colorization_1/summary
```

## Summary
Dataset used in this project is hand-picked portraits from Helen and WIKI datasets with train_size=2150 and test_size=100.
Model was trained for 65 epochs(about 1 hour on Floydhub cloud with Nvidia Tesla K80).
