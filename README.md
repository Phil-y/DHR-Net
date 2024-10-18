# MCS_Net

This repository is the official implementation of DHR-Net is Definitely What You Need for
Medical Image Segmentation using PyTorch.

![DHR-Net](Fig/DHR-Net.png)



## Main Environments

- python 3.9
- pytorch 2.1.0
- torchvision 0.16.0



## Requirements

Install from the `requirements.txt` using:

```
pip install -r requirements.txt
```



## Prepare the dataset.

- The DDTI datasets, can be found [here](https://drive.google.com/drive/folders/1za9f38XKx-VYPxxb_xx83Dpk-Wg3Yaw8?usp=sharing)
- The Kvasir-Seg  datasets, can be found [here](https://link.zhihu.com/?target=https%3A//datasets.simula.no/downloads/kvasir-seg.zip)
- The  ISIC2018 datasets, can be found [here](https://challenge.isic-archive.com/data/)
- The Synapse datasets, can be found [here](https://drive.google.com/drive/folders/1Vofe2TSVry0FZYLNisvPKvR_67aSj0ml?usp=sharing)




## Train the Model

First, modify the model, dataset and training hyperparameters (including learning rate, batch size img size and optimizer etc) in `Config.py`

Then simply run the training code.

```
python3 train_model.py
```



## Evaluate the Model

#### 2. Test the Model

Please make sure the right model, dataset and hyperparameters setting  is selected in `Config.py`. 

Then change the test_session in `Config.py` .

Then simply run the evaluation code.

```
python3 test_model.py
```



## Reference

- [TransUNet](https://github.com/Beckschen/TransUNet)
- [UCTransNet](https://github.com/McGregorWwww/UCTransNet)




## Citation

If you find this work useful in your research or use this dataset in your work, please consider citing the following papers:




