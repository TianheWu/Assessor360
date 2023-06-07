import logging
import torch
import numpy as np
import os


def split_dataset_JUFE(dis_path, split_seed=20):
    np.random.seed(split_seed)
    dis_data = {}
    index = []
    for root, ds, fs in os.walk(dis_path):
        for f in fs:
            temp_list = f.split('_')
            if temp_list[0] not in index:
                index.append(temp_list[0])
            dis_data[f] = temp_list[0]
    
    np.random.shuffle(index)
    np.random.seed(20)

    l = len(index)
    train_name_idx = index[:round(l * 0.8)]
    val_name_idx = index[round(l * 0.8):]

    train_name = []
    val_name = []
    
    for key, values in dis_data.items():
        if values in train_name_idx:
            train_name.append(key)
        if values in val_name_idx:
            val_name.append(key)

    return train_name, val_name


def split_dataset_mvaqd(ref_dis_score_path, split_seed=20):
    np.random.seed(split_seed)
    dis_data = {}
    ref_data = []
    with open(ref_dis_score_path, 'r') as list_file:
        for line in list_file:
            ref, dis, score = line.split()
            dis_data[dis] = ref
            if ref not in ref_data:
                ref_data.append(ref)

    np.random.shuffle(ref_data)
    np.random.seed(20)

    l = len(ref_data)
    train_name_idx = ref_data[:round(l * 0.8)]
    val_name_idx = ref_data[round(l * 0.8):]

    train_name = []
    val_name = []
    
    for key, values in dis_data.items():
        if values in train_name_idx:
            train_name.append(key)
        if values in val_name_idx:
            val_name.append(key)

    return train_name, val_name


def split_dataset_oiqa(ref_label_path, dis_label_path, split_seed=20):
    np.random.seed(split_seed)
    dis_data = {}
    val_idx = 1
    count = 0
    with open(dis_label_path, 'r') as list_file:
        for line in list_file:
            count += 1
            dis, score = line.split()
            if count == 21:
                val_idx += 1
                count = 1
            dis_data[dis] = val_idx
    
    ref_data = []
    with open(ref_label_path, 'r') as list_file:
        for line in list_file:
            ref, score = line.split()
            if ref not in ref_data:
                ref_data.append(int(ref))
    
    np.random.shuffle(ref_data)
    np.random.seed(20)

    l = len(ref_data)
    train_name_idx = ref_data[:round(l * 0.8)]
    val_name_idx = ref_data[round(l * 0.8):]

    # print(train_name_idx)

    train_name = []
    val_name = []
    
    for key, values in dis_data.items():
        if values in train_name_idx:
            train_name.append(key)
        if values in val_name_idx:
            val_name.append(key)

    return train_name, val_name


def split_dataset_iqaodi(ref_dis_score_path, split_seed=20):
    np.random.seed(split_seed)
    dis_data = {}
    with open(ref_dis_score_path, 'r') as list_file:
        for line in list_file:
            ref, dis, score = line.split()
            dis_data[dis] = ref

    ref_data = []
    with open(ref_dis_score_path, 'r') as list_file:
        for line in list_file:
            ref, dis, score = line.split()
            if ref not in ref_data:
                ref_data.append(ref)
    
    np.random.shuffle(ref_data)
    np.random.seed(20)

    l = len(ref_data)
    train_name_idx = ref_data[:round(l * 0.8)]
    val_name_idx = ref_data[round(l * 0.8):]

    train_name = []
    val_name = []
    
    for key, values in dis_data.items():
        if values in train_name_idx:
            train_name.append(key)
        if values in val_name_idx:
            val_name.append(key)

    return train_name, val_name


def split_dataset_cviqd(ref_label_path, dis_label_path, split_seed=20):
    np.random.seed(split_seed)
    dis_data = {}
    val_idx = 0
    with open(dis_label_path, 'r') as list_file:
        for line in list_file:
            dis, score = line.split()
            if int(dis[:3]) % 34 == 1:
                val_idx += 34
            dis_data[int(dis[:3])] = val_idx

    ref_data = []
    with open(ref_label_path, 'r') as list_file:
        for line in list_file:
            ref, score = line.split()
            ref_data.append(int(ref[:3]))
    
    np.random.shuffle(ref_data)
    np.random.seed(20)

    l = len(ref_data)
    train_name_idx = ref_data[:round(l * 0.8)]
    val_name_idx = ref_data[round(l * 0.8):]

    train_name = []
    val_name = []
    
    for key, values in dis_data.items():
        if values in train_name_idx:
            train_name.append(key)
        if values in val_name_idx:
            val_name.append(key)
    
    logging.info("Train name: {}".format(train_name))
    logging.info("Val name: {}".format(val_name))

    return train_name, val_name


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        d1 = sample['d_img_org']
        score = sample['score']
        name = sample['name']
        range_val = sample['range_val']
        d1 = (d1 - self.mean) / self.var

        if 'r_img_ref' in sample.keys():
            r1 = sample['r_img_ref']
            r1 = (r1 - self.mean) / self.var
            sample = {'d_img_org': d1, 'r_img_ref': r1, 'score': score, 'name': name, 'range_val': range_val}
        else:
            sample = {'d_img_org': d1, 'score': score, 'name': name, 'range_val': range_val}

        return sample


class RandHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d1 = sample['d_img_org']
        score = sample['score']
        name = sample['name']
        range_val = sample['range_val']
        prob_lr = np.random.random()

        if 'r_img_ref' in sample.keys():
            r1 = sample['r_img_ref']
            if prob_lr > 0.5:
                d1 = np.fliplr(d1).copy()
                r1 = np.fliplr(r1).copy()
            sample = {'d_img_org': d1, 'r_img_ref': r1, 'score': score, 'name': name, 'range_val': range_val}
        else:
            if prob_lr > 0.5:
                d1 = np.fliplr(d1).copy()
            sample = {'d_img_org': d1, 'score': score, 'name': name, 'range_val': range_val}

        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d1 = sample['d_img_org']
        score = sample['score']
        name = sample['name']
        range_val = sample['range_val']
        d1 = torch.from_numpy(d1).type(torch.FloatTensor)
        score = torch.from_numpy(score).type(torch.FloatTensor)

        if 'r_img_ref' in sample.keys():
            r1 = sample['r_img_ref']
            r1 = torch.from_numpy(r1).type(torch.FloatTensor)
            sample = {'d_img_org': d1, 'r_img_ref': r1, 'score': score, 'name': name, 'range_val': range_val}
        else:
            sample = {'d_img_org': d1, 'score': score, 'name': name, 'range_val': range_val}
        
        return sample
