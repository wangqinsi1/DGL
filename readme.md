# DGL: Device Generic Latency model for Neural Architecture Search

This is  PyTorch implementation for  [2023-TMC-DGL: Device Generic Latency model for Neural Architecture Search](https://ieeexplore.ieee.org/document/10042973).

### Prerequisites

------

- Python 3.7.6
- PyTorch 1.9.0
- CUDA 11.1

### Installation

------

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install nas-bench-201
```

## 1.Experiment on NAS-Bench-201 Search Space

------

We provide code that can reproduce the main experimental results on the NAS-Bench-201 search space. 

#### Train the device generic latency model

Using the following settings you can train the device generic latency predictor and save the trained model . We provide the network architectures used for training and their network parameters.

```
$ python main.py --search_space 'nasbench201' \
                 --mode 'train' \
                 --train_devices 'Galaxy A9,Honor 9C,iQOO 5 Pro,Samsung Galaxy A80,Samsung Galaxy A90 5G,Samsung Galaxy M30s,Samsung Galaxy S20+ 5G,Samsung Galaxy Note10,Xiaomi Redmi 10X Pro 5G,Xiaomi Redmi K20 Pro,Honor V30 Pro,VIVO NEX'\
                 --epoch 50000 \
                 --lr 1e-1
```

The trained model will be saved in

```
./result/nasbench201/pretrainmodel.pth
```

We provide a trained model and you can test it directly.

#### Test the latency model on unseen devices

With the following settings you can calculate the correlation coefficient R, the root mean square error RMSE, and the mean absolute error MAE between the predicted latencies and the actual latencies on unseen devices.

```
$ python main.py --search_space 'nasbench201' \
		 --mode 'test' \
         --load_path './result/nasbench201/pretrainmodel.pt' \
         --data_path './data/nasbench201/' \
		 --test_devices 'Huawei Mate 20,Samsung Galaxy A10s,Xiaomi Redmi 5 Plus,Sony Xperia XZ2,Motorola Moto E6s'         
```

The predicted latencies will be saved in

```
./result/nasbench201/[test_device]_timepre.txt
```

You can replace the tested device by saving the device hardware parameters and the measured latencies in

```
./data/nasbench201/hardware/[test_device].txt
./data/nasbench201/latency/[test_device].txt
```

#### Efficient Latency-constrained NAS

In the NAS-Bench-201 search space, we provide an oracle accuracy predictor combined with our latency predictor for latency-constrained NAS search. You can get the accuracy and predicted latency of the searched network.

```
$ python main.py --search_space 'nasbench201' \
                 --mode 'nas' \
                 --load_path './result/nasbench201/pretrainmodel.pt' \
                 --nas_target_device 'Samsung Galaxy A10s' \
                 --latency_constraint 40 
```

The searched network is saved in

```
./result/nasbench201/nas/[nas_target_device]_[latency_constraint]_model.pt
```

You can change the target device by saving the device parameters in

```
./data/nasbench201/NAS/[nas_target_device].txt
```

## 2. Experiment on FBNetV3 Search Space

------

We also provide training, testing, and NAS on the FBNetV3 search space. For the NAS process, we use the DARS predictor and genetic algorithm search in FBNetV3.

#### Train the device generic latency model

You can train the predictor with the following settings.

```
$ python main.py --search_space 'fbnetv3' \
                 --mode 'train' \
                 --train_devices 'Galaxy A9,Honor 9C,iQOO 5 Pro,Samsung Galaxy A80,Samsung Galaxy A90 5G,Samsung Galaxy M30s,Samsung Galaxy S20+ 5G,Samsung Galaxy Note10,Xiaomi Redmi 10X Pro 5G,Xiaomi Redmi K20 Pro,Honor V30 Pro,VIVO NEX'\
                 --epoch 50000 \
                 --lr 1e-1
```

We also provide trained model in

```
./result/fbnetv3/pretrainmodel.pth
```

#### Test the latency model on unseen devices

You can get the R, RMSE, and MAE of the predicted and the measured latencies on the test devices.

```
$ python main.py --search_space 'fbnetv3' \
		 --mode 'test' \
         --load_path './result/fbnetv3/pretrainmodel.pt' \
         --data_path './data/fbnetv3/' \
		 --test_devices 'Asus ROG Phone 3,Oppo A5,Xiaomi Mi 8 Lite,Samsung Galaxy S8+'         
```

The predicted latencies will be saved in

```
./result/fbnetv3/[test_device]_timepre.txt
```

#### Latency-limited NAS combined with NARS predictor and genetic search	

We provide the NARS accuracy predictor and genetic search algorithm in FBNetV3 for NAS. You can get the predicted accuracy and latency of the searched network by the following settings.

```
$ python main.py --search_space 'fbnetv3' \
                 --mode 'nas' \
                 --load_path './result/fbnetv3/pretrainmodel.pt' \
                 --nas_target_device 'Asus R0G Ph0ne 3' \
                 --latency_constraint 90 \
                 --POPULATION_SIZE 100 \
                 --EPOCH 100 \
                 --CROSSOVER_RATE 2 \
                 --GENE_LENGTH 28 \
                 --MUTATION_RATE 0.05
```

The searched network is saved in

```
./result/fbnetv3/nas/[nas_target_device]_[latency_constraint]_model.pt
```

You can change the target device by saving the device parameters in

```
./data/fbnetv3/NAS/[nas_target_device].txt
```

