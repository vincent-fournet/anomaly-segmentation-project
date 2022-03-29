from cv2 import threshold
import torch
import torch.nn as nn
import torch.nn.functional as F
from feature import Extractor
from torch.utils.data import DataLoader
import torch.optim as optim
from unet import UNet, UNetAE

import time
import datetime
import os
import random 
import math

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
from skimage import measure
from skimage.transform import resize
import pandas as pd

from feat_cae import FeatCAE

import joblib
from sklearn.decomposition import PCA

from utils import *


class AnoSegUNET():
    """
    Anomaly segmentation model: DFR.
    """
    def __init__(self, cfg):
        super(AnoSegUNET, self).__init__()
        self.cfg = cfg
        self.path = cfg.save_path    # model and results saving path

        self.cutout_sizes = [2, 4, 8, 16]
        self.num_disjoint_masks = 3


        self.log_step = 10
        self.data_name = cfg.data_name

        self.img_size = cfg.img_size
        self.threshold = cfg.thred
        self.device = torch.device(cfg.device)

        # datasest
        self.train_data_path = cfg.train_data_path
        self.test_data_path = cfg.test_data_path
        self.train_data = self.build_dataset(is_train=True)
        self.test_data = self.build_dataset(is_train=False)

        # dataloader
        self.train_data_loader = DataLoader(self.train_data, batch_size=4, shuffle=True, num_workers=4)
        self.test_data_loader = DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=1)


        # saving paths
        self.model_name = cfg.model_name
        print("model name:", self.model_name)

        # optimizer
        if self.model_name == "unet":
            self.model_name = "{}{}".format(self.model_name, self.cfg.unet_size)
            self.unet = UNetAE(latent_size=self.cfg.unet_size)
            self.lr = 0.0001
            self.optimizer = optim.Adam(self.unet.parameters(), lr=self.lr, weight_decay=0)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.97)
            self.loss_fn = nn.MSELoss()

        elif self.model_name == "riad":
            self.unet = UNet()
            self.lr = 0.0001
            self.optimizer = optim.Adam(self.unet.parameters(), lr=self.lr, weight_decay=0)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.97)
            self.loss_fn = nn.MSELoss() #self.riad_loss_fn

        self.unet.to(cfg.device)


        self.subpath = self.data_name + "/" + self.model_name
        self.model_path = os.path.join(self.path, "models/" + self.subpath + "/model")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.eval_path = os.path.join(self.path, "models/" + self.subpath + "/eval")
        if not os.path.exists(self.eval_path):
            os.makedirs(self.eval_path)

    def build_dataset(self, is_train):
        from MVTec import NormalDataset, TestDataset
        normal_data_path = self.train_data_path
        abnormal_data_path = self.test_data_path
        if is_train:
            dataset = NormalDataset(normal_data_path, normalize=True)
        else:
            dataset = TestDataset(path=abnormal_data_path, normalize=True)
        return dataset

    def train(self):

        start_time = time.time()
        iters_per_epoch = len(self.train_data_loader)  # total iterations every epoch

        # train
        epochs = 100  # total epochs
        for epoch in range(1, epochs+1):
            self.unet.train()
            losses = []
            for i, normal_img in enumerate(self.train_data_loader):
                normal_img = normal_img.to(self.device)
                # forward and backward
                total_loss = self.optimize_step(normal_img, i)

                # statistics and logging
                loss = {}
                loss['total_loss'] = total_loss.data.item()
                
                # tracking loss
                losses.append(loss['total_loss'])

            self.scheduler.step()
            
            if epoch % 1 == 0:
                #                 self.save_model()

                print('Epoch {}/{}'.format(epoch, epochs))
                print('-' * 10)
                elapsed = time.time() - start_time
                total_time = ((epochs * iters_per_epoch) - (epoch * iters_per_epoch + i)) * elapsed / (
                        epoch * iters_per_epoch + i + 1)
                epoch_time = (iters_per_epoch - i) * elapsed / (epoch * iters_per_epoch + i + 1)

                epoch_time = str(datetime.timedelta(seconds=epoch_time))
                total_time = str(datetime.timedelta(seconds=total_time))
                elapsed = str(datetime.timedelta(seconds=elapsed))

                log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}]".format(
                    elapsed, epoch_time, total_time, epoch, epochs, i + 1, iters_per_epoch)

                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

            if epoch % 10 == 0:
                # save model
                self.save_model()
                self.validation(epoch)

#             print("Cost total time {}s".format(time.time() - start_time))
#             print("Done.")
            self.tracking_loss(epoch, np.mean(np.array(losses)))

        # save model
        self.save_model()
        print("Cost total time {}s".format(time.time() - start_time))
        print("Done.")

    def optimize_step(self, input_data, step):
        self.optimizer.zero_grad()

        if "unet" in self.model_name:
            dec = self.unet(input_data)
        elif self.model_name == "riad":
            cutout_size = random.choice(self.cutout_sizes)
            dec = self.inpainting_reconstruct(input_data, cutout_size)

        # loss
        total_loss = self.loss_fn(dec, input_data)

        # self.reset_grad()
        total_loss.backward()

        self.optimizer.step()

        return total_loss

    def inpainting_reconstruct(self, mb_img, cutout_size):
        _, _, h, w = mb_img.shape
        disjoint_masks = self._create_disjoint_masks((h, w), cutout_size, self.num_disjoint_masks)

        mb_reconst = 0
        for mask in disjoint_masks:
            mb_cutout = mb_img * mask
            mb_inpaint = self.unet(mb_cutout)
            mb_reconst += mb_inpaint * (1 - mask)
        return mb_reconst

    def _create_disjoint_masks(self,img_size,cutout_size=8,num_disjoint_masks=3):
        img_h, img_w = img_size
        grid_h = math.ceil(img_h / cutout_size)
        grid_w = math.ceil(img_w / cutout_size)
        num_grids = grid_h * grid_w
        disjoint_masks = []
        for grid_ids in np.array_split(np.random.permutation(num_grids), num_disjoint_masks):
            flatten_mask = np.ones(num_grids)
            flatten_mask[grid_ids] = 0
            mask = flatten_mask.reshape((grid_h, grid_w))
            mask = mask.repeat(cutout_size, axis=0).repeat(cutout_size, axis=1)
            mask = torch.tensor(mask, requires_grad=False, dtype=torch.float)
            mask = mask.to(self.device)
            disjoint_masks.append(mask)

        return disjoint_masks


    def save_model(self, epoch=0):
        # save model weights
        torch.save({'unet': self.unet.state_dict()},
                   os.path.join(self.model_path, 'unet.pth'))

    def validation(self, epoch):
        return 0

    def tracking_loss(self, epoch, loss):
        out_file = os.path.join(self.eval_path, '{}_epoch_loss.csv'.format(self.model_name))
        if not os.path.exists(out_file):
            with open(out_file, mode='w') as f:
                f.write("Epoch" + ",loss" + "\n")
        with open(out_file, mode='a+') as f:
            f.write(str(epoch) + "," + str(loss) + "\n")

    def load_model(self, path=None):
        print("Loading model...")
        if path is None:
            model_path = os.path.join(self.model_path, 'unet.pth')
            print("model path:", model_path)
            if not os.path.exists(model_path):
                print("Model not exists.")
                return False

            if torch.cuda.is_available():
                data = torch.load(model_path)
            else:
                data = torch.load(model_path,
                                  map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU, using a function

            self.unet.load_state_dict(data['unet'])
            print("Model loaded:", model_path)
        return True


    def eval_segmentation(self):
        if self.load_model():
            print("Model Loaded.")
        else:
            print("None pretrained models.")
            return

        self.unet.eval()
        print("Computing segmentation threshold")
        anomaly_scores_good = []
        n_good_examples = 0

        anomaly_maps_test = []
        masks_test = []
        n_test_examples = 0
        for i, (img, mask, name) in enumerate(self.test_data_loader):  # batch size is 1.
            mask = mask.squeeze().numpy()

            with torch.no_grad():
                if "unet" in self.model_name:
                    pred = self.unet(img)
                    anomaly_map = (img-pred)**2
                    anomaly_map = anomaly_map.squeeze().numpy()
                    anomaly_map = np.mean(anomaly_map, axis=0)

                elif self.model_name == "riad":
                    preds = [self.inpainting_reconstruct(img, c) for c in [2,4,8,16]]
                    anomaly_maps = [(img-pred)**2 for pred in preds]
                    anomaly_maps = [np.mean(anomaly_map.squeeze().numpy(), axis=0) for anomaly_map in anomaly_maps]
                    anomaly_map = np.mean(anomaly_maps, axis=0)


            if "good" in name[0]: # good examples to select the threshold
                anomaly_scores_good.append(anomaly_map.flatten().tolist())
                n_good_examples += 1
            else:
                anomaly_maps_test.append(anomaly_map)
                masks_test.append(mask.astype(bool))
                n_test_examples += 1
        
        seg_thresh_95 = np.percentile(anomaly_scores_good,95)
        seg_thresh_99 = np.percentile(anomaly_scores_good,99)
        print("\n Segmentation threshold @ 95 : ", seg_thresh_95)
        print("\n Segmentation threshold @ 99 : ", seg_thresh_99)

        IoU_95 = 0
        IoU_99 = 0
        for anomap, mask in zip(anomaly_maps_test, masks_test):
            segmap_95 = anomap > seg_thresh_95
            segmap_99 = anomap > seg_thresh_99
            IoU_95 += np.sum(np.logical_and(segmap_95, mask)) / np.sum(np.logical_or(segmap_95, mask))
            IoU_99 += np.sum(np.logical_and(segmap_99, mask)) / np.sum(np.logical_or(segmap_99, mask))
        print("\n IoU @ 95 : ", IoU_95 / n_test_examples)
        print("\n IoU @ 99 : ", IoU_99 / n_test_examples)

        seg_auc_score = roc_auc_score(np.array(masks_test).ravel(), np.array(anomaly_maps_test).ravel())
        print("ROC AUC Score : ", seg_auc_score)

    def reconstruction_test(self, expect_fpr=0.3, max_step=5000):
        from torchvision.utils import save_image

        if self.load_model():
            print("Model Loaded.")
        else:
            print("None pretrained models.")
            return
        self.unet.eval()
        print("Calculating AUC, IOU, PRO metrics on testing data...")
        masks = []
        scores = []
        avg_recons_error = 0
        n_examples = 0
        for i, (img, mask, name) in enumerate(self.test_data_loader):  # batch size is 1.
            # data
            mask = mask.squeeze().numpy()

            # anomaly score
            # anomaly_map = self.score(img).data.cpu().numpy()
            if "other" not in name[0]:
                continue
            n_examples += 1
            if n_examples > 5:
                return

            with torch.no_grad():
                if "unet" in self.model_name:
                    preds= [self.unet(img)]
                elif self.model_name == "riad":
                    preds = [self.inpainting_reconstruct(img, c) for c in [2,4,8,16]]
                #error = self.loss_fn(pred, img)
            #avg_recons_error += error
            
            if self.cfg.save_reconstruction:
                for pred_i,pred in enumerate(preds):
                    pred = torch.squeeze(pred.detach())

                    save_name = name[0].replace('\\data','\\{}_reconstruction'.format(self.model_name)).replace('\\test','')
                    if not os.path.exists(save_name[:-7]):
                        os.makedirs(save_name[:-7])

                    pred = pred - torch.min(pred)
                    pred = pred / torch.max(pred)
                    save_image(pred, save_name[:-4]+"_recons{}.png".format(pred_i))

                img1 = img - torch.min(img)
                img1 = img1 / torch.max(img1)
                save_image(img1, save_name[:-4]+"_orig.png")
                masks.append(mask)
        #print("Reconstruction error : ", avg_recons_error / n_examples)
