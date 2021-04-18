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
textList1 = ['在读书的时候，中考无疑是人生的转折点，但不管它是怎么转的，转向何方，后面的路都还是留给你们自己走的。不要有太大的压力，因为人生的可能性有很多的，也许下一秒你的世界依旧会是彩色，塞翁失马焉知非福。武汉华中艺术学校是一所全日制的艺术职业高中，开设有美术、音乐(声乐、钢琴)、舞蹈(体育舞蹈、民族舞)三个方向，均可参加高考和艺术高考。你们要知道参加高考考上一所终身受益的大学，远比你在培训机构或者中职技校里混三年直接出来就业的强太多。职业高中是没有把握考上普通高中退而求其次的选择，因为青春只有一次，没有经历过高考的人生是不完美的。武汉华中艺术学校让中考失利的你，再次有高考圆梦大学的机会', '“7号就要参加高考了，可自己的身份证丢失了”，7月5日上午9时许，一考生急匆匆跑进万山公安分局下溪派出所户籍室，对户籍民警说道。\n\n原来，这名考生姓杨，万山下溪人，在铜仁市第八中学就读高三，眼看就要参加高考了，可在医院就医的时候不慎将身份证丢失，所以赶紧来派出所补办身份证。得知情况后，派出所立即为其开通绿色通道办理身份证明，仅短短10分钟左右，民警为考生杨某当场办理了临时身份证，确保了考生能顺利参加高考，当考生杨某拿到临时身份证时，对民警表示感谢。\n\n警方提醒:临近高考，请广大考生一定要小心，保护好相关资料，谨防丢失。', '“美国新冠肺炎患者跳海致海盐污染”是谣言!(转载)', '印度首都新德里监狱系统累计有221人感染新冠肺炎', '印度首都新德里监狱系统累计有221人感染新冠肺炎', '强证据表明羟氯喹预防新冠不比安慰剂更有效', '《自然》子刊:新冠病毒谱系可能已在蝙蝠中传播数十年', '高考防疫家长怎么做？国家卫健委发布10条关键提示', '重磅!北大合作团队新冠强效药研发有新进展(转载)', '关于制定新冠病毒核酸检测收费标准的通知-安顺市发展和改革委员会（安顺市粮食局）', '台湾新增5例新冠肺炎确诊病例-新华网']

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

def get_split_text(text, split_len=32, overlap_len=8):
    split_text=[]
    if(len(text)//split_len==0):
        split_text.append(text)
    for w in range(len(text)//split_len):
        if w == 0:   #第一次,直接分割长度放进去
            text_piece = text[:split_len]
        else:      # 否则, 按照(分割长度-overlap)往后走
            window = split_len - overlap_len
            text_piece = text[w * window: w * window +split_len]
        split_text.append(text_piece)
    return split_text

def predict(textList):
    key = []
    value = []
    newTextList = []
    for i in range(len(textList)):
        tmp = []
        tmp.append(i)
        key.append(i)
        value.append(tmp)
    listmap = dict(zip(key,value))
    for i in range(len(textList)):
        tmpList = get_split_text(textList[i])
        newTextList.extend(tmpList)
        listmap[i]=len(tmpList)
    print(listmap)
    print(newTextList)
    test_data = load_dataset(newTextList, config.pad_size)
    test_iter = build_iterator(test_data, config)
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, lables in test_iter:
            outputs = model(texts)
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predict)
    new_predict_all = []
    index = 0
    for i in range(len(key)):
        tmpPredict = []
        num = listmap[i]
        for j in range(num):
            tmpPredict.append(predict_all[index])
            index = index+1
        new_predict_all.append(max(tmpPredict, key=tmpPredict .count))
    return new_predict_all

def listToJson(lst):
    keys = [str(x) for x in np.arange(len(lst))]
    list_json = dict(zip(keys, lst))
    str_json = json.dumps(list_json, indent=2, ensure_ascii=False)  # json转为string
    return str_json
class IndexHandler(tornado.web.RequestHandler):
    def post(self, *args, **kwargs):
        j = json.loads(self.request.body.decode('utf-8'))
        # print(j["textList"])
        textList = j["textList"]

        result = predict(textList)
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

    print(predict(textList1))
    #main()
