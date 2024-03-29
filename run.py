# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network,test
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train 这是训练模块
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)

    #test 这是测试模块，在训练完后如果不想跑测试集就把这段注释掉，要跑测试集就不要注释
    #test(config, model, test_iter)
    # model.load_state_dict(torch.load(config.save_path))
    # model.eval()
    # with torch.no_grad():
    #     for texts, lables in test_iter:
    #         print("label:")
    #         print(lables)
    #
    #         outputs = model(texts)
    #         predict = torch.max(outputs.data, 1)[1].cpu().numpy()
    #         print("predict")
    #         print(predict)