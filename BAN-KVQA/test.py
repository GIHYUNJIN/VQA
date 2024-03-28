"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import sys
import os
import glob
import json
import time
import argparse
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import KvqaFeatureDataset, Dictionary
import base_model
import utils
from registry import dictionary_dict
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=512)
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--q_emb', type=str, default='bertrnn', choices=dictionary_dict.keys())
    parser.add_argument('--op', type=str, default='')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--gamma', type=int, default=8)
    parser.add_argument('--split', type=str, default='test') #
    parser.add_argument('--input', type=str, default='saved_models')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--on_do_q', action='store_true', help='turn on dropout of question embedding?')
    parser.add_argument('--finetune_q', action='store_true', help='finetune question embedding?')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--logits', action='store_true')
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    return args

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def get_question_answer(q_id, dataloader):
    test_data = read_json('data/test_qa_anno.json')
    ques = []
    ans = []
    for j in test_data:
        if j['question_id'] == q_id:
            ques.append(j['question'])
            ans.append(j['answers'][0]['answer'])
    ques = ' '.join(ques)
    ans = ' '.join(ans)
    return ques, ans

def get_answer(q_id, real_data):
    for j in real_data:
        if j['question_id'] == q_id:
            return j['answers'][0]['answer']

def get_answer_topk(p, dataloader):
    answer_list = []
    _v, idx = torch.topk(p, 2)        
    for i in idx:
        answer_list.append(dataloader.dataset.label2ans[i.item()])
    return answer_list

@torch.no_grad()
def get_logits_topk(model, dataloader, logger):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    pred = torch.FloatTensor(N, M).zero_()
    qIds = []
    idx = 0
    now_time = 'time : ' + time.strftime('%Y-%m-%d %H:%M:%S')
    logger.write(now_time)
    
    bar = progressbar.ProgressBar(max_value=N).start()
    
    for v, b, q, i, _, _ in iter(dataloader):
        bar.update(idx)
                
        batch_size = v.size(0)
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        logits, _, att = model(v, b, q, None, None)
        pred[idx:idx+batch_size,:].copy_(logits.data)
        qIds[idx:idx+batch_size] = i
        idx += batch_size

    bar.finish()
    return pred, qIds

def top_k_accuracy(results):
     
    pred_answer = np.array([i['pred_answer'] for i in results])
    real_answer = np.array([i['real_answer'] for i in results])
    
    correct_topk = np.array([true_label in predicted_labels for true_label, predicted_labels in zip(real_answer, pred_answer)])
    
    topk_accuracy = np.mean(correct_topk)
    
    return topk_accuracy

def make_json(logits, qIds, dataloader):
    real_data = read_json('data/test_qa_anno.json')
    question_id_index = [i['question_id'] for i in real_data]
    
    utils.assert_eq(logits.size(0), len(qIds))
    
    results = []
    for i, v in enumerate(range(logits.size(0))):
        result = {}
        result['question_id'] = qIds[v]
        result['question'] = real_data[question_id_index.index(qIds[v])]['question']
        result['real_answer'] = get_answer(qIds[v], real_data)
        result['pred_answer'] = get_answer_topk(logits[v], dataloader)
        results.append(result)

    return results

def process(args, model, eval_loader, logger):
    model_path = glob.glob(os.path.join(args.input, '*.pth'))[0]

    logger.write('loading %s' % model_path)
    model_data = torch.load(model_path)

    model = nn.DataParallel(model).cuda()
    model.load_state_dict(model_data.get('model_state', model_data))

    model.train(False)

    logits, qIds = get_logits_topk(model, eval_loader, logger)
    results = make_json(logits, qIds, eval_loader)

    acc = top_k_accuracy(results)
    
    if args.debug == True:
        for i in results:
            # print(i)
            question_id = i['question_id']
            question = i['question']
            real_answer = i['real_answer']
            pred_answer = i['pred_answer']
            logger.write(f'qestion ID : {question_id}\tquestion : {question}\treal_answer : {real_answer}\tpred_answer : {pred_answer}')

    logger.write(f'Top-2 Accuracy : {acc}')
    
    utils.create_dir(args.output)
    
    with open(args.output+'/%s_result.json' \
        % args.split, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == '__main__':
    
    args = parse_args()
    
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    command_line = ' '.join(sys.argv)
    logger.write(f'python3 {command_line}')
    logger.write(args.__repr__())

    torch.backends.cudnn.benchmark = True

    if 'bert' in args.q_emb:
        dictionary = None
    else:
        dictionary_path = os.path.join(args.dataroot, dictionary_dict[args.q_emb]['dict'])
        dictionary = Dictionary.load_from_file(dictionary_path)

    eval_dset = KvqaFeatureDataset(args.split, dictionary, tokenizer=dictionary_dict[args.q_emb]['tokenizer'])

    logger.write(f'num of data : {len(eval_dset.entries)}')

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid, args.op, args.gamma,
                                             args.q_emb, args.on_do_q, args.finetune_q).cuda()
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)

    process(args, model, eval_loader, logger)
    now_time = 'time : ' + time.strftime('%Y-%m-%d %H:%M:%S')
    logger.write(now_time)