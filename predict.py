# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network,test
from tqdm import tqdm
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import logging
logging.basicConfig(level=logging.INFO)

# parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
# args = parser.parse_args()
PAD, CLS ,SEP= '[PAD]', '[CLS]' ,'[SEP]' # padding符号, bert中综合信息符号

def load_dataset(path, pad_size=32):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):

            lin1 = line.strip().strip('\t')
            lin = lin1.replace('\n', '')
            if not lin:
                continue
            # if lin.find('\t')>0:
            #     content, label = lin.split('\t')
            # else:
            #     content = lin
            #     label = -1
            content = lin
            label = -1
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token + [SEP]
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)


            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * (pad_size)
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            # print("token_ids", token_ids)
            # print("token_ids len", len(token_ids))
            contents.append((token_ids, int(label), seq_len, mask))
    return contents

def predict(textList):
    return None

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    #model_name = args.model  # bert
    x = import_module('models.' + 'bert')
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    test_data = load_dataset(config.test_path, config.pad_size)
    print("test_data_size:",len(test_data))
    test_iter = build_iterator(test_data, config)
    # print(test_iter)
    print("test_iter_size:", len(test_iter))
    time_dif = get_time_dif(start_time)


    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    print("Time usage:", time_dif)
    # train(config, model, train_iter, dev_iter, test_iter)
    #test(config,model,test_iter)
    print("start predict...")
    start_time = time.time()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, lables in test_iter:
            # print("label:")
            # print(lables)

            outputs = model(texts)
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            # print("predict")
            print("predict",predict)
            predict_all = np.append(predict_all, predict)
    print("predict_all",predict_all)
    print("predict_all_size:",len(predict_all))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)