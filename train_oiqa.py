import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from models.assessor360 import creat_model
from config import Config
from utils_tools.process_oiqa import ToTensor, RandHorizontalFlip, Normalize
from utils_tools.process_oiqa import split_dataset_cviqd, split_dataset_iqaodi, split_dataset_oiqa, split_dataset_mvaqd
from utils_tools.process_oiqa import split_dataset_JUFE

from torch.utils.tensorboard import SummaryWriter 
from load_train import train_oiqa, eval_oiqa


os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(config):
    if not os.path.exists(config.log_path): 
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        "dataset_name": "iqaodi",

        # dataset path
        "oiqa_train_dis_path": "/mnt/data_16TB/wth22/IQA_dataset/OIQA/distorted_images/",
        "cviqd_train_dis_path": "/mnt/data_16TB/wth22/IQA_dataset/CVIQ_database/CVIQ/",
        "iqaodi_train_dis_path": "/mnt/data_16TB/wth22/IQA_dataset/IQA-ODI/all_ref_test_img/",
        "mvaqd_train_dis_path": "/mnt/data_16TB/wth22/IQA_dataset/MVAQD-dataset/",
        "JUFE_dataset_path": "/mnt/cpath2/lf2/OIQA_dataset/Fang2022_dis/",
        "JUFE_user_data_path": "/mnt/cpath2/lf2/OIQA_dataset/HMData/",
        
        # label
        "oiqa_dis_label": "./data/OIQA/OIQA_dis_label.txt",
        "oiqa_ref_label": "./data/OIQA/OIQA_ref_label.txt",
        "iqaodi_dis_label": "./data/IQA_ODI/iqaodi_ref_dis_label.txt",
        "cviqd_dis_label": "./data/cviqd/CVIQD_dis_label.txt",
        "cviqd_ref_label": "./data/cviqd/CVIQD_ref_label.txt",
        "mvaqd_dis_label": "./data/MVAQD/MVAQD_ref_dis_label.txt",
        "JUFE_label_path": "./data/JUFE/JUFE_label.xls",

        # optimization
        "batch_size": 4,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 300,
        "val_freq": 1,
        "num_workers": 16,
        "split_seed": 0,

        # model
        "num_layers": 6,
        "viewport_nums": 5,
        "embed_dim": 128,
        "dab_layers": 4,

        # data
        "start_points": [[0, 0], [0, 0], [0, 0]],
        "viewport_size": (224, 224),
        "fov": [110, 110],
        "model_weight_path": None,

        # load & save checkpoint
        "model_name": "exp1-iqaodi_seed0",
        "type_name": "iqaodi",
        "ckpt_path": "./output/models/",               # directory for saving checkpoint
        "log_path": "./output/log/",
        "tensorboard_path": "./output/tensorboard/"
    })

    config.log_file = config.model_name + ".log"
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.type_name, config.model_name)
    config.ckpt_path = os.path.join(config.ckpt_path, config.type_name, config.model_name)
    config.log_path = os.path.join(config.log_path, config.type_name)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    
    if not os.path.exists(config.tensorboard_path):
        os.makedirs(config.tensorboard_path)

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)

    if config.dataset_name == 'cviqd':
        from data.cviqd.cviqd_label import CVIQD
        train_name, val_name = split_dataset_cviqd(config.cviqd_ref_label, config.cviqd_dis_label, split_seed=config.split_seed)
        dis_train_path = config.cviqd_train_dis_path
        dis_val_path = config.cviqd_train_dis_path
        label_train_path = config.cviqd_dis_label
        label_val_path = config.cviqd_dis_label
        Dataset = CVIQD
    elif config.dataset_name == 'iqaodi':
        from data.IQA_ODI.iqaodi_label import IQAODI
        train_name, val_name = split_dataset_iqaodi(config.iqaodi_dis_label, split_seed=config.split_seed)
        dis_train_path = config.iqaodi_train_dis_path
        dis_val_path = config.iqaodi_train_dis_path
        label_train_path = config.iqaodi_dis_label
        label_val_path = config.iqaodi_dis_label
        Dataset = IQAODI
    elif config.dataset_name == 'oiqa':
        from data.OIQA.oiqa_label import OIQA
        train_name, val_name = split_dataset_oiqa(config.oiqa_ref_label, config.oiqa_dis_label, split_seed=config.split_seed)
        dis_train_path = config.oiqa_train_dis_path
        dis_val_path = config.oiqa_train_dis_path
        label_train_path = config.oiqa_dis_label
        label_val_path = config.oiqa_dis_label
        Dataset = OIQA
    elif config.dataset_name == 'mvaqd':
        from data.MVAQD.mvaqd_label import MVAQD
        train_name, val_name = split_dataset_mvaqd(config.mvaqd_dis_label, split_seed=config.split_seed)
        dis_train_path = config.mvaqd_train_dis_path
        dis_val_path = config.mvaqd_train_dis_path
        label_train_path = config.mvaqd_dis_label
        label_val_path = config.mvaqd_dis_label
        Dataset = MVAQD
    elif config.dataset_name == 'jufe':
        from data.JUFE.jufe import JUFE
        train_name, val_name = split_dataset_JUFE(config.JUFE_dataset_path, split_seed=config.split_seed)
        dis_train_path = config.JUFE_dataset_path
        dis_val_path = config.JUFE_dataset_path
        label_train_path = config.JUFE_label_path
        label_val_path = config.JUFE_label_path
        Dataset = JUFE
    else:
        raise ValueError("No dataset, you need to add this new dataset.")


    # data load
    train_dataset = Dataset(
        dis_path=dis_train_path,
        txt_file_name=label_train_path,
        list_name=train_name,
        transform=transforms.Compose([Normalize(0.5, 0.5), RandHorizontalFlip(), ToTensor()]),
        viewport_size=config.viewport_size,
        viewport_nums=config.viewport_nums,
        fov=config.fov,
        start_points=config.start_points
    )
    val_dataset = Dataset(
        dis_path=dis_val_path,
        txt_file_name=label_val_path,
        list_name=val_name,
        transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
        viewport_size=config.viewport_size,
        viewport_nums=config.viewport_nums,
        fov=config.fov,
        start_points=config.start_points
    )

    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(val_dataset)))
    logging.info('train : val ratio is: {:.4}'.format(len(train_dataset) / len(val_dataset)))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False
    )

    net = creat_model(config=config, pretrained=False)
    net = nn.DataParallel(net).cuda()

    logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

    # loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # train & validation
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    main_score = 0

    for epoch in range(0, config.n_epoch):
        # visual(net, val_loader)
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, rho_s, rho_p = train_oiqa(epoch, net, criterion, optimizer, train_loader)

        writer.add_scalar("Train_loss", loss_val, epoch)
        writer.add_scalar("Train_SRCC", rho_s, epoch)
        writer.add_scalar("Train_PLCC", rho_p, epoch)

        if (epoch + 1) % config.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running val {} in epoch {}'.format(config.dataset_name, epoch + 1))
            loss, rho_s, rho_p = eval_oiqa(config, epoch, net, criterion, val_loader)
            logging.info('Eval done...')

            writer.add_scalar("Val_loss", loss, epoch)
            writer.add_scalar("Val_SRCC", rho_s, epoch)
            writer.add_scalar("Val_PLCC", rho_p, epoch)

            if rho_s + rho_p > main_score:
                main_score = rho_s + rho_p
                logging.info('======================================================================================')
                logging.info('============================== best main score is {} ================================='.format(main_score))
                logging.info('======================================================================================')
        
                best_srocc = rho_s
                best_plcc = rho_p

                # save weights
                ckpt_name = "best_ckpt.pt"
                model_save_path = os.path.join(config.ckpt_path, ckpt_name)
                torch.save(net.module.state_dict(), model_save_path)
                logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch + 1, best_srocc, best_plcc))
        
        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))