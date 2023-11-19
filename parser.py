####################################################################################################
# DGL: Device Generic Latency model for Neural Architecture Search
####################################################################################################
import argparse
def str2list(v):
    if isinstance(v, list):
        return v
    else:
        return [item for item in v.split(',')]

    
def get_parser():
    parser = argparse.ArgumentParser()
    # General Settings
    parser.add_argument('--gpu', type=str, default='0', help='set visible gpus')
    parser.add_argument('--seed', type=int, default=3, help='set seed')
    parser.add_argument('--mode', type=str, default='test', help='train|test|nas')
    parser.add_argument('--main_path', type=str, default='.')
    parser.add_argument('--search_space', type=str, default='nasbench201', help='fbnetv3|nasbench201')
    parser.add_argument('--load_path', type=str, default='./result/nasbench201/pretrainmodel.pt', help='model checkpoint path')    
    # Data 
    parser.add_argument('--data_path', type=str, default='./data/nasbench201/', help='model checkpoint path')    
    parser.add_argument('--train_devices', type=str2list, 
                default='Asus R0G Phone 3,Galaxy A9,Honor 9C,iQOO 5 Pro,Samsung Galaxy A80,Samsung Galaxy A90 5G,Samsung Galaxy M30s,Samsung Galaxy S20+ 5G,Samsung Galaxy Note10,Xiaomi Redmi 10X Pro 5G,Xiaomi Redmi K20 Pro,Honor V30 Pro,VIVO NEX')
    parser.add_argument('--test_devices', type=str2list, 
                default='Huawei Mate 20,Samsung Galaxy A10s,Xiaomi Redmi 5 Plus,Sony Xperia XZ2,Motorola Moto E6s',help='Asus ROG Phone 3,Oppo A5,Xiaomi Mi 8 Lite,Samsung Galaxy S8+')
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate")
    parser.add_argument('--epoch', type=float, default=50000, help="training epoch")
    # Save Result
    parser.add_argument('--save_path', type=str, default='results', help='')
    # Latency-constrainted NAS
    parser.add_argument('--nas_target_device', type=str, default='Samsung Galaxy A10s', help="target device of NAS process")
    parser.add_argument('--latency_constraint', type=float, default=40, help="latency constraint when performing NAS process")
    parser.add_argument('--POPULATION_SIZE', type=int, default=100, help="population size of genetic search process")
    parser.add_argument('--EPOCH', type=int, default=10, help="number of evolution iterations during NAS search")
    parser.add_argument('--CROSSOVER_RATE', type=float, default=2, help="population crossover rate during NAS search")
    parser.add_argument('--GENE_LENGTH', type=int, default=28, help="population gene count during NAS search")
    parser.add_argument('--MUTATION_RATE', type=float, default=0.05, help="population mutation rate during NAS search")
    args = parser.parse_args([])
    return args

