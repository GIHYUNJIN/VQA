import os
import glob
import argparse
import json
import pickle
import pandas as pd
import numpy as np
import csv
import sys
import base64
import cv2
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from imageio import imread

csv.field_size_limit(sys.maxsize)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/root/BAN-KVQA/data/')
    args = parser.parse_args()
    return args

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file)
        
def trans_qa_data(data_path, split, save_file_name):
    anno = read_json(data_path + '2.최종산출물/1.데이터/' + split + '/2.라벨링데이터/LABEL/질의응답/AI Hub 업로드/질의응답.json')
    
    trans_qa = []
    for data in anno:
        for qa_data in data['QA_list']:
            qa_dict = {}
            if qa_data['answer_type'] == 'answer':
                qa_dict['question_id'] = qa_data['QA_ID']
                qa_dict['image'] = data['Scene_Graph_ID'] + '.jpg'
                qa_dict['image_id'] = data['Scene_Graph_ID']
                qa_dict['source'] = 'kvqa'
                qa_dict['answers'] = [{
                    'answer' : qa_data['answer'],
                    'answer_confidence' : 'yes',
                    'answer_id' : 1,
                }]
                qa_dict['question'] = qa_data['question']
                qa_dict['answerable'] = 1
                qa_dict['answer_type'] = 'other'
                trans_qa.append(qa_dict)
    save_json(data_path + save_file_name, trans_qa)
    
def trans_bbox_data(data_path, split, save_file_name):
    anno = read_json(data_path + '2.최종산출물/1.데이터/' + split + '/2.라벨링데이터/LABEL/장면그래프/AI Hub 업로드/장면그래프.json')
    
    print(f'{split} file num: {len(anno)}')
    
    trans_anno = []
    for i in anno:
        trans_dict = {}
        trans_dict['image_file_path'] = data_path + '2.최종산출물/1.데이터/' + split + '/1.원천데이터/' + i['Category'] + '/' + i['Scene_Graph_ID'] + '.jpg'
        trans_dict['image_id'] = i['Scene_Graph_ID']
        label_list = []
        label_name_list = []
        score_list = []
        bbox_list = []
        
        if i['OBJECTS'] == []:
            pass
        else:
            for obj in i['OBJECTS']:
                label_list.append(obj['OBJECT_ID'])
                label_name_list.append(obj['NAME'])
                score_list.append(1.0)
                bbox_list.append(
                    [
                        obj['X'],
                        obj['Y'],
                        obj['W'] + obj['X'],
                        obj['H'] + obj['Y'],
                    ]
                )
            
            trans_dict['label_ids'] = label_list
            trans_dict['label'] = label_name_list
            trans_dict['score'] = score_list
            trans_dict['bbox'] = bbox_list
            trans_anno.append(trans_dict)
    save_json(data_path + save_file_name, trans_anno)
    
    
if __name__ == '__main__':
    
    args = parse_args()
    
    args.data_path
    
    # make bbox annotation
    trans_bbox_data(args.data_path, '1.Training', 'train_object_anno.json')
    trans_bbox_data(args.data_path, '2.Validation', 'val_object_anno.json')
    trans_bbox_data(args.data_path, '3.Test', 'test_object_anno.json')
    
    # make QA annotation
    trans_qa_data(args.data_path, '1.Training', 'train_qa_anno.json')
    trans_qa_data(args.data_path, '2.Validation', 'val_qa_anno.json')
    trans_qa_data(args.data_path, '3.Test', 'test_qa_anno.json')