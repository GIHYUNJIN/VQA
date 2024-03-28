"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import glob
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import KvqaFeatureDataset, Dictionary
import base_model
import utils
from registry import dictionary_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=512)
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--q_emb', type=str, default='glove-rg', choices=dictionary_dict.keys())
    parser.add_argument('--op', type=str, default='')
    parser.add_argument('--gamma', type=int, default=8, help='glimpse')
    parser.add_argument('--use_both', action='store_true', help='use both train/val datasets to train?')
    parser.add_argument('--finetune_q', action='store_true', help='finetune question embedding?')
    parser.add_argument('--on_do_q', action='store_true', help='turn on dropout of question embedding?')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='saved_models/ban-kvqa')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1204, help='random seed')
    parser.add_argument('--dataset', type=str, default='test', help='dataset type')
    args = parser.parse_args()
    return args

def compute_topk_score_with_logits(logits, labels, k=2):
    labels = labels * (10/3)
    logits = torch.topk(logits, 2, dim=1)[1]
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits, 1)
    scores = (one_hots * labels)
    return scores

def calc_entropy(att): # size(att) = [b x g x v x q]
    sizes = att.size()
    p = att.view(-1, sizes[1], sizes[2] * sizes[3])
    return (-p * (p + utils.EPS).log()).sum(2).sum(0) # g

@torch.no_grad()
def inference(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    dset = dataloader.dataset
    # n_answer_type = torch.zeros(len(dset.idx2type))
    # score_answer_type = torch.zeros(len(dset.idx2type))
    entropy = None
    if hasattr(model.module, 'glimpse'):
        entropy = torch.Tensor(model.module.glimpse).zero_().cuda()

    for i, (v, b, q, a, c, at, _) in enumerate(dataloader): # q_id - ghjin
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        a = a.cuda()
        c = c.cuda().unsqueeze(-1).float()
        at = at.cuda()
        # answer_type = torch.zeros(v.size(0), len(dset.idx2type)).cuda()
        # answer_type.scatter_(1, at.unsqueeze(1), 1)

        pred, conf, att = model(v, b, q, a, c)
        batch_score = compute_topk_score_with_logits(pred, a.data)
        # type_score = batch_score.sum(-1, keepdim=True) * answer_type
        batch_score = batch_score.sum()
        score += batch_score.item()

        # n_answer_type += answer_type.sum(0).cpu()
        # score_answer_type += type_score.sum(0).cpu()

        upper_bound += (a.max(1)[0]).sum().item()
        num_data += pred.size(0)
        if att is not None and 0 < model.module.glimpse:
            entropy += calc_entropy(att.data)[:model.module.glimpse]

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)

    return score, upper_bound, entropy, n_answer_type, score_answer_type

if __name__ == '__main__':
    args = parse_args()

    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'eval_log.txt'))
    logger.write(args.__repr__())

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.input is None:
        args.input = args.output


    if 'bert' in args.q_emb:
        dictionary = None
    else:
        dictionary_path = os.path.join(args.dataroot, dictionary_dict[args.q_emb]['dict'])
        dictionary = Dictionary.load_from_file(dictionary_path)
    # val_dset = KvqaFeatureDataset('val', dictionary, tokenizer=dictionary_dict[args.q_emb]['tokenizer'])
    val_dset = KvqaFeatureDataset(args.dataset, dictionary, tokenizer=dictionary_dict[args.q_emb]['tokenizer'])

    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(val_dset, args.num_hid, args.op, args.gamma,
                                             args.q_emb, args.on_do_q, args.finetune_q).cuda()

    model = nn.DataParallel(model).cuda()

    optim = None
    epoch = 0

    # load snapshot
    if args.input is not None:
        path = os.path.join(args.output)
        print('loading %s' % path)

        model_data = torch.load(glob.glob(os.path.join(path, "model_epoch*.pth"))[-1])
        model.load_state_dict(model_data.get('model_state', model_data))
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        epoch = model_data['epoch'] + 1

    eval_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)

    model.train(False)
    val_score, bound, entropy, val_n_type, val_type_score = inference(model, eval_loader)

    # logger.write('\nMean val upper bound: {}'.format(bound))
    # logger.write('\nMean val score: {}'.format(val_score))
    # logger.write('\nAnswer type: '+', '.join(val_dset.idx2type))
    # logger.write('\n'+'Number of examples for each type on val: {}'.format(val_n_type))
    # logger.write('\n'+'Mean score for each type on val: {}'.format(val_type_score / val_n_type))

    logger.write('\nMean score: {}'.format(val_score))
    logger.write('\nAnswer type: '+', '.join(val_dset.idx2type))
    logger.write('\n'+'Number of examples for each type on val: {}'.format(val_n_type.item()))
    logger.write('\n'+'Mean score for each type on val: {}'.format((val_type_score / val_n_type).item()))