import os
import torch
import numpy as np
import cv2
import math
import xlrd

from utils_tools.erp2rec import ERP2REC



np.random.seed(2)


def get_random_scanpath():
    x = np.random.randint(-90, 90, size=(1, ))
    y = np.random.randint(-180, 180, size=(1, ))
    random_coordes = np.dstack((x, y)).squeeze(0)
    return random_coordes


def normal_distribution(x, mean, sigma):
    return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)


def generate_lat_prob():
    x = np.linspace(-90, 90, 181) / 90
    y = {}
    val = []

    for i in range(x.shape[0]):
        val.append(normal_distribution(x[i], 0, 0.2))

    val = np.array(val)
    val = softmax(val)

    idx = -90
    for i in range(val.shape[0]):
        y[idx] = val[i]
        idx += 1

    return y


def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_head_data(path):
    track_list = []
    teachessheets = xlrd.open_workbook(path)
    table = teachessheets.sheet_by_index(0)
    lines = table.nrows
    for i in range(0, lines):
        pitch_yaw = [table.cell(i, 0).value, table.cell(i, 1).value]
        track_list.append(pitch_yaw)
    return np.array(track_list)


def get_img_path(path):
    img_list = []
    for root, ds, fs in os.walk(path):
        for f in fs:
            full_name = os.path.join(root, f)
            img_list.append(full_name)
    img_list = np.array(img_list)
    return img_list


def get_file_path(path):
    """
        Distinguish the bad and good starting point
    """
    all_sp  = []
    for root, ds, fs in os.walk(path):
        temp_split = root.split('/')
        if temp_split[-1] not in ['A', 'B', 'C', 'D', "bad", "good"]:
            for f in fs:
                full_name = os.path.join(root, f)
                all_sp.append(full_name)

    all_sp = np.array(all_sp)
    return all_sp


def get_img_sps(img_name, all_sp):
    img_sp = []
    for i in range(all_sp.shape[0]):
        # need to refine to '/'
        split_info = all_sp[i].split('/')
        if img_name == split_info[-1][:-4]:
            img_sp.append(all_sp[i])
    img_sp = np.array(img_sp)

    sp_list = []
    for i in range(img_sp.shape[0]):
        sp_list.append(get_head_data(img_sp[i]))
    
    sp_list = np.array(sp_list)
    return sp_list


def get_label(path):
    teachessheets = xlrd.open_workbook(path)
    table = teachessheets.sheet_by_index(0)
    lines = table.nrows

    score_list = []
    for i in range(0, lines):
        score_info = [
            str(table.cell(i, 0).value),
            str(table.cell(i, 1).value),
            str(table.cell(i, 2).value),
            str(table.cell(i, 6).value)
        ]
        score_list.append(score_info)
    return score_list


class JUFE(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, viewport_size,
                 viewport_nums, fov, start_points):
        super(JUFE, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform
        self.viewport_size = viewport_size
        self.viewport_nums = viewport_nums
        self.fov = fov
        self.domain_transform = ERP2REC()
        self.start_points = start_points
        self.user_data_path = "/mnt/OIQA_dataset/HMData/good/" # your HMData path

        self.second_limit = "15s"
        self.start_point_label = "good"

        dis_files_data, score_data = [], []
        label_list = get_label(self.txt_file_name)

        for i in range(1, len(label_list)):
            if label_list[i][1] in list_name:
                session = label_list[i][0].split('_')[1]
                if session == self.start_point_label and label_list[i][3] == self.second_limit: 
                    dis_files_data.append(label_list[i][1])
                    score_data.append(float(label_list[i][2]))

        # define the latitude weights
        self.lat_weights = generate_lat_prob()

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data, range_val = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)

        self.range_val = range_val
        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)

        # prepare the GT data
        all_sp = get_file_path(self.user_data_path)
        sp_list = get_img_sps(img_name=d_img_name[:-4], all_sp=all_sp) # num_scanpaths, lat, lon

        len_viewports = sp_list.shape[1]

        start_points = []
        scanpath_GT = []

        if self.second_limit == "5s":
            sp_list = sp_list[:, :int(len_viewports / 3), :]
        
        num_seq = sp_list.shape[0]
        if num_seq > 15:
            num_seq = 15
        for i in range(num_seq):
            start_points.append([int(sp_list[i][0][0]), int(sp_list[i][0][1])])
            seq = []
            for j in range(sp_list.shape[1]):
                if j % 10 == 0:
                    seq.append(sp_list[i][j])
            scanpath_GT.append(seq)
        scanpath_GT = np.array(scanpath_GT)
        
        # use GT
        d1 = []
        for i in range(scanpath_GT.shape[0]):
            seq_list = []
            for j in range(scanpath_GT.shape[1]):
                r, c = scanpath_GT[i][j][0], scanpath_GT[i][j][1]
                viewport = self.domain_transform.toREC(
                    frame=d_img,
                    center_point=np.array([c, r]),
                    FOV=self.fov,
                    width=self.viewport_size[0],
                    height=self.viewport_size[1]
                )
                seq_list.append(viewport)
            d1.append(np.array(seq_list, dtype=np.float32))
        d1 = np.array(d1)

        # use ours
        # self.start_points = start_points[:3]
        # d1 = self.select_viewports(d_img)

        M, N, H, W, C = d1.shape

        for i in range(d1.shape[0]):
            for j in range(d1.shape[1]):
                d1[i][j] = cv2.cvtColor(d1[i][j], cv2.COLOR_BGR2RGB)
                d1[i][j] = np.array(d1[i][j]).astype('float32') / 255
        d1 = np.transpose(d1, (0, 1, 4, 2, 3))
        score = self.data_dict['score_list'][idx]
        d1 = np.array(d1)
        sample = {
            'd_img_org': d1,
            'score': score,
            'name': d_img_name,
            'range_val': self.range_val,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range, range

    def cal_entropy(self, sig):
        len = sig.size
        sig_set = list(set(sig))
        p_list = [np.size(sig[sig == i]) / len for i in sig_set]
        entropy = np.sum([p * np.log2(1.0 / p) for p in p_list])
        return entropy

    def get_img_entropy(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        sum_entropy = self.cal_entropy(gray_img.flatten())
        return sum_entropy
    
    def cal_next_patch_coordinate(self, viewport, cur_coordinate=(0, 0)):
        """
            c: Longitude [-180, 180], r: Latitude [-90, 90]
        """
        r, c = cur_coordinate[0], cur_coordinate[1]
        idx2next_coordinate = {0:(c - 24, r + 24), 1:(c + 0, r + 24), 2:(c + 24, r + 24),
                               3:(c - 24, r + 0),  4:(c + 24, r + 0), 5:(c - 24, r - 24),
                               6:(c - 0, r - 24),  7:(c + 24, r - 24)}
        
        # obtain viewport shape
        H, W, C = viewport.shape
        patch_size = H // 4

        ent_list = []
        for i in range(0, H - patch_size, patch_size):
            for j in range(0, W - patch_size, patch_size):
                if i == patch_size and j == patch_size:
                    continue
                img_patch = viewport[i:i + patch_size * 2, j:j + patch_size * 2, :]
                
                # calculate the image entropy
                cur_ent = self.get_img_entropy(img_patch)
                ent_list.append(cur_ent)

        ent_list = np.array(ent_list)
        ent_list = softmax(ent_list)

        # get current latitude weights
        cur_lat_weights = []
        zero_idx = np.ones(8)
        for i in range(8):
            cur_c, cur_r = idx2next_coordinate[i]
            cur_c = self.modify_c(cur_c)
            if cur_r > 90 or cur_r < -90:
                cur_lat_weights.append(-1e-9)
                zero_idx[i] = 0
            else:
                cur_lat_weights.append(self.lat_weights[cur_r])

            if str(cur_r) + str(cur_c) in self.vis_coords.keys():
                cur_lat_weights[i] *= 0.7

        cur_lat_weights = softmax(np.array(cur_lat_weights) * 100)
        cur_lat_weights = cur_lat_weights * zero_idx
        integrated_weights = softmax(ent_list * cur_lat_weights * 100)

        np.random.seed(20)
        while True:
            idx = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], p=integrated_weights.ravel())
            if zero_idx[idx] != 0:
                break

        # new longitude and latitude
        _c, _r = idx2next_coordinate[idx]
        _c = self.modify_c(_c)

        if _r > 90 or _r < -90 or _c > 180 or _c < -180:
            print("Warning:::::::==================::::::::: {}".format(_r))
            print("Warning:::::::==================::::::::: {}".format(_c))

        return (_r, _c)

    def dfs_get_viewport(self, img, cur_coordinate=(0, 0)):
        # r:[90, -90], c:[-180, 180]
        r, c = cur_coordinate[0], cur_coordinate[1]
        viewport = self.domain_transform.toREC(
            frame=img,
            center_point=np.array([c, r]),
            FOV=self.fov,
            width=self.viewport_size[0],
            height=self.viewport_size[1]
        )

        self.vis_coords[str(r) + str(c)] = 1
        self.viewports_list.append(viewport)
        if len(self.viewports_list) == self.viewport_nums:
            return

        next_coordinate = self.cal_next_patch_coordinate(viewport, (r, c))
        next_r, next_c = next_coordinate
        return self.dfs_get_viewport(img, cur_coordinate=(next_r, next_c))
        
    def select_viewports(self, img):
        self.seq_list = []
        for i in range(len(self.start_points)):
            self.viewports_list = []
            self.vis_coords = {}
            self.dfs_get_viewport(img, cur_coordinate=self.start_points[i])
            self.seq_list.append(np.array(self.viewports_list, dtype=np.float32))
        return np.array(self.seq_list)
    
    def modify_c(self, _c):
        if _c > 180:
            _c = -180 + (_c - 180)
        elif _c < -180:
            _c = 180 - (-180 - _c)
        else:
            pass
        return _c