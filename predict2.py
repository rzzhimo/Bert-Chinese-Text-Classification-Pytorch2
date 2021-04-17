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
import urllib
import tornado.httpserver
import tornado.ioloop
import tornado.options
from tornado.web import RequestHandler
import py_eureka_client.eureka_client as eureka_client
from tornado.options import define, options
import py_eureka_client.netint_utils as netint_utils
from time import sleep
import json
define("port", default=3333, help="run on the given port", type=int)

logging.basicConfig(level=logging.INFO)

PAD, CLS ,SEP= '[PAD]', '[CLS]' ,'[SEP]' # padding符号, bert中综合信息符号

def load_dataset(textList, pad_size=32):
    contents = []
    for line in textList:
        lin1 = line.strip().strip('\t')
        lin = lin1.replace('\n', '')
        if not lin:
            continue
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
    test_data = load_dataset(textList, config.pad_size)
    test_iter = build_iterator(test_data, config)
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, lables in test_iter:
            outputs = model(texts)
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predict)
    return predict_all

def listToJson(lst):
    keys = [str(x) for x in np.arange(len(lst))]
    list_json = dict(zip(keys, lst))
    str_json = json.dumps(list_json, indent=2, ensure_ascii=False)  # json转为string
    return str_json
class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        textList = self.get_argument('textList')
        #print(textList)
        json_re = json.loads(textList)
        print(json_re)
        result = predict(json_re)
        predict_all = []
        for i in range(len(result)):
            predict_all.append(class1.get(result[i]))
        result1 = listToJson(predict_all)
        print(''.join(result1))
        self.write(''.join(result1))


def main():
    tornado.options.parse_command_line()
    # 注册eureka服务
    eureka_client.init(eureka_server="http://localhost:9000/eureka/",
                                       app_name="python-service",
                                       instance_port=3333)
    app = tornado.web.Application(handlers=[(r"/predict", IndexHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    # model_name = args.model  # bert
    x = import_module('models.' + 'bert')
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    class1 = {0: '财经', 1: '房产', 2: '股票',3:'教育',4:'科技',5:'社会',6:'时政',7:'体育',8:'游戏',9:'娱乐'}
    main()
