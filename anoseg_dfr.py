import torch
import torch.nn as nn
import torch.nn.functional as F
#from extractors.feature import Extractor
from feature import Extractor
from torch.utils.data import DataLoader
import torch.optim as optim
#from data.MVTec import NormalDataset, TrainTestDataset

import time
import datetime
import os
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


class AnoSegDFR():
    """
    Anomaly segmentation model: DFR.
    """
    def __init__(self, cfg):
        super(AnoSegDFR, self).__init__()
        self.cfg = cfg
        self.path = cfg.save_path    # model and results saving path

        self.n_layers = len(cfg.cnn_layers)
        self.n_dim = cfg.latent_dim

        self.log_step = 10
        self.data_name = cfg.data_name

        self.img_size = cfg.img_size
        self.threshold = cfg.thred
        self.device = torch.device(cfg.device)

        # feature extractor
        self.extractor = Extractor(backbone=cfg.backbone,
                 cnn_layers=cfg.cnn_layers,
                 upsample=cfg.upsample,
                 is_agg=cfg.is_agg,
                 kernel_size=cfg.kernel_size,
                 stride=cfg.stride,
                 dilation=cfg.dilation,
                 featmap_size=cfg.featmap_size,
                 device=cfg.device).to(self.device)

        # datasest
        self.train_data_path = cfg.train_data_path
        self.test_data_path = cfg.test_data_path
        self.train_data = self.build_dataset(is_train=True)
        self.test_data = self.build_dataset(is_train=False)

        # dataloader
        self.train_data_loader = DataLoader(self.train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
        self.test_data_loader = DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=1)
        self.eval_data_loader = DataLoader(self.train_data, batch_size=10, shuffle=False, num_workers=2)


        # autoencoder classifier
        self.autoencoder, self.model_name = self.build_classifier()
        if cfg.model_name != "":
            self.model_name = cfg.model_name
        print("model name:", self.model_name)

        # optimizer
        self.lr = cfg.lr
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.lr, weight_decay=0)

        # saving paths
        self.subpath = self.data_name + "/" + self.model_name
        self.model_path = os.path.join(self.path, "models/" + self.subpath + "/model")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.eval_path = os.path.join(self.path, "models/" + self.subpath + "/eval")
        if not os.path.exists(self.eval_path):
            os.makedirs(self.eval_path)

    def build_classifier(self):
        # self.load_dim(self.model_path)
        if self.n_dim is None:
            print("Estimating one class classifier AE parameter...")
            feats = torch.Tensor()
            for i, normal_img in enumerate(self.eval_data_loader):
                i += 1
                if i > 1:
                    break
                normal_img = normal_img.to(self.device)
                feat = self.extractor.feat_vec(normal_img)
                feats = torch.cat([feats, feat.cpu()], dim=0)
            # to numpy
            feats = feats.detach().numpy()
            # estimate parameters for mlp
            pca = PCA(n_components=0.90)    # 0.9 here try 0.8
            pca.fit(feats)
            n_dim, in_feat = pca.components_.shape
            print("AE Parameter (in_feat, n_dim): ({}, {})".format(in_feat, n_dim))
            self.n_dim = n_dim
        else:
            for i, normal_img in enumerate(self.eval_data_loader):
                i += 1
                if i > 1:
                    break
                normal_img = normal_img.to(self.device)
                feat = self.extractor.feat_vec(normal_img)
            in_feat = feat.shape[1]

        print("BN?:", self.cfg.is_bn)
        autoencoder = FeatCAE(in_channels=in_feat, latent_dim=self.n_dim, is_bn=self.cfg.is_bn).to(self.device)
        model_name = "AnoSegDFR({})_{}_l{}_d{}_s{}_k{}_{}".format('BN' if self.cfg.is_bn else 'noBN',
                                                                self.cfg.backbone, self.n_layers,
                                                                self.n_dim, self.cfg.stride[0],
                                                                self.cfg.kernel_size[0], self.cfg.upsample)

        return autoencoder, model_name

    def build_dataset(self, is_train):
        from MVTec import NormalDataset, TestDataset
        normal_data_path = self.train_data_path
        abnormal_data_path = self.test_data_path
        if is_train:
            dataset = NormalDataset(normal_data_path, normalize=True)
        else:
            dataset = TestDataset(path=abnormal_data_path)
        return dataset

    def train(self):
        if self.load_model():
            print("Model Loaded.")
            return

        start_time = time.time()

        # train
        iters_per_epoch = len(self.train_data_loader)  # total iterations every epoch
        epochs = self.cfg.epochs  # total epochs
        for epoch in range(1, epochs+1):
            self.extractor.train()
            self.autoencoder.train()
            losses = []
            for i, normal_img in enumerate(self.train_data_loader):
                normal_img = normal_img.to(self.device)
                # forward and backward
                total_loss = self.optimize_step(normal_img)

                # statistics and logging
                loss = {}
                loss['total_loss'] = total_loss.data.item()
                
                # tracking loss
                losses.append(loss['total_loss'])
            
            if epoch % 5 == 0:
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

    def tracking_loss(self, epoch, loss):
        out_file = os.path.join(self.eval_path, '{}_epoch_loss.csv'.format(self.model_name))
        if not os.path.exists(out_file):
            with open(out_file, mode='w') as f:
                f.write("Epoch" + ",loss" + "\n")
        with open(out_file, mode='a+') as f:
            f.write(str(epoch) + "," + str(loss) + "\n")

    def optimize_step(self, input_data):
        self.extractor.train()
        self.autoencoder.train()

        self.optimizer.zero_grad()

        # forward
        input_data = self.extractor(input_data)

        # print(input_data.size())
        dec = self.autoencoder(input_data)

        # loss
        total_loss = self.autoencoder.loss_function(dec, input_data.detach().data)

        # self.reset_grad()
        total_loss.backward()

        self.optimizer.step()

        return total_loss

    def save_model(self, epoch=0):
        # save model weights
        torch.save({'autoencoder': self.autoencoder.state_dict()},
                   os.path.join(self.model_path, 'autoencoder.pth'))
        np.save(os.path.join(self.model_path, 'n_dim.npy'), self.n_dim)

    def load_model(self, path=None):
        print("Loading model...")
        if path is None:
            model_path = os.path.join(self.model_path, 'autoencoder.pth')
            print("model path:", model_path)
            if not os.path.exists(model_path):
                print("Model not exists.")
                return False

            if torch.cuda.is_available():
                data = torch.load(model_path)
            else:
                data = torch.load(model_path,
                                  map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU, using a function

            self.autoencoder.load_state_dict(data['autoencoder'])
            print("Model loaded:", model_path)
        return True

    def score(self, input, save = False):
        """
        Args:
            input: image with size of (img_size_h, img_size_w, channels)
        Returns:
            score map with shape (img_size_h, img_size_w)
        """
        self.extractor.eval()
        self.autoencoder.eval()

        input = self.extractor(input)
        dec = self.autoencoder(input)

        # sample energy
        scores = self.autoencoder.compute_energy(dec, input, save = save)
        scores = scores.reshape((1, 1, self.extractor.out_size[0], self.extractor.out_size[1]))    # test batch size is 1.
        scores = nn.functional.interpolate(scores, size=self.img_size, mode="bilinear", align_corners=True).squeeze()
        # print("score shape:", scores.shape)
        return scores

    def eval_segmentation(self):

        if self.load_model():
            print("Model Loaded.")
        else:
            print("None pretrained models.")
            return

        print("Computing segmentation threshold")
        anomaly_scores_good = [[],[],[],[]]
        n_good_examples = 0


        anomaly_maps_test = []
        masks_test = []
        n_test_examples = 0
        for i, (img, mask, name) in enumerate(self.test_data_loader):  # batch size is 1.
            # data
            img = img.to(self.device)
            mask = mask.squeeze().numpy()

            anomaly_map = self.score(img).data.cpu().numpy().squeeze()
            anomap1 = np.mean(anomaly_map[:128], axis=0)
            anomap2 = np.mean(anomaly_map[128:384], axis=0)
            anomap3 = np.mean(anomaly_map[384:1408], axis=0)
            anomap4 = np.mean(anomaly_map[1408:3456], axis=0)

            
            if "good" in name: # good examples to select the threshold
                anomaly_scores_good[0].append(anomap1.flatten().tolist())
                anomaly_scores_good[1].append(anomap2.flatten().tolist())
                anomaly_scores_good[2].append(anomap3.flatten().tolist())
                anomaly_scores_good[3].append(anomap4.flatten().tolist())
                n_good_examples += 1
            else:
                anomaly_maps_test.append([anomap1,anomap2,anomap3,anomap4])
                masks_test.append(mask.astype(bool))
                n_test_examples += 1

        anomaly_scores_good = np.array(anomaly_scores_good)
        print(anomaly_scores_good.shape, anomaly_scores_good.dtype)
        m1, s1 = np.mean(anomaly_scores_good[0]), np.std(anomaly_scores_good[0])
        m2, s2 = np.mean(anomaly_scores_good[1]), np.std(anomaly_scores_good[1])
        m3, s3 = np.mean(anomaly_scores_good[2]), np.std(anomaly_scores_good[2])
        m4, s4 = np.mean(anomaly_scores_good[3]), np.std(anomaly_scores_good[3])
        anomaly_scores_good[0] = (anomaly_scores_good[0] - m1) / s1
        anomaly_scores_good[1] = (anomaly_scores_good[1] - m2) / s2
        anomaly_scores_good[2] = (anomaly_scores_good[2] - m3) / s3
        anomaly_scores_good[3] = (anomaly_scores_good[3] - m4) / s4
        print("Mean and sd of each scale : \nScale 1 : {}, {} \nScale2 : {}, {} \nScale3 : {}, {} \nScale4 : {}, {}".format(m1,s1,m2,s2,m3,s3,m4,s4))

        anomaly_scores_good = np.sum(anomaly_scores_good, axis=0)

        seg_thresh_95 = np.percentile(anomaly_scores_good,95)
        seg_thresh_99 = np.percentile(anomaly_scores_good,99)
        print("\n Segmentation threshold @ 95 : ", seg_thresh_95)
        print("\n Segmentation threshold @ 99 : ", seg_thresh_99)

        final_anomaly_maps_test = []

        IoU_95 = 0
        IoU_99 = 0
        for (anomap1,anomap2,anomap3,anomap4), mask in zip(anomaly_maps_test, masks_test):
            anomap1 = torch.tensor(anomap1)
            anomap2 = torch.tensor(anomap2)
            anomap3 = torch.tensor(anomap3)
            anomap4 = torch.tensor(anomap4)
            anomap = (anomap1 - m1)/s1 + (anomap2 - m2)/s2 + (anomap3 - m3)/s3 + (anomap4 - m4)/s4
            anomap = anomap.reshape((1, 1, self.extractor.out_size[0], self.extractor.out_size[1]))    # test batch size is 1.
            anomap = nn.functional.interpolate(anomap, size=self.img_size, mode="bilinear", align_corners=True).squeeze()
                
            anomap = anomap.numpy().squeeze()
            final_anomaly_maps_test.append(anomap)
            segmap_95 = anomap > seg_thresh_95
            segmap_99 = anomap > seg_thresh_99
            IoU_95 += np.sum(np.logical_and(segmap_95, mask)) / np.sum(np.logical_or(segmap_95, mask))
            IoU_99 += np.sum(np.logical_and(segmap_99, mask)) / np.sum(np.logical_or(segmap_99, mask))

        print("\n IoU @ 95 : ", IoU_95 / n_test_examples)
        print("\n IoU @ 99 : ", IoU_99 / n_test_examples)
        seg_auc_score = roc_auc_score(np.array(masks_test).ravel(), np.array(final_anomaly_maps_test).ravel())
        print("ROC AUC Score : ", seg_auc_score)


    def eval_segmentation_original(self):
        # Segmentation evaluation, when averaging all the anomaly maps
        if self.load_model():
            print("Model Loaded.")
        else:
            print("None pretrained models.")
            return

        print("Computing segmentation threshold")
        anomaly_scores_good = []
        n_good_examples = 0


        anomaly_maps_test = []
        masks_test = []
        n_test_examples = 0
        for i, (img, mask, name) in enumerate(self.test_data_loader):  # batch size is 1.
            # data
            img = img.to(self.device)
            mask = mask.squeeze().numpy()
            anomaly_map = self.score(img).data.cpu().numpy().squeeze()
            anomap = np.mean(anomaly_map, axis=0)
            
            if "good" in name: # good examples to select the threshold
                anomaly_scores_good.append(anomap.flatten().tolist())
                n_good_examples += 1
            else:
                anomap = torch.tensor(anomap)
                anomap = anomap.reshape((1, 1, self.extractor.out_size[0], self.extractor.out_size[1]))    # test batch size is 1.
                anomap = nn.functional.interpolate(anomap, size=self.img_size, mode="bilinear", align_corners=True).squeeze()
                anomap = anomap.numpy().squeeze()
                anomaly_maps_test.append(anomap)
                masks_test.append(mask.astype(bool))
                n_test_examples += 1

        anomaly_scores_good = np.array(anomaly_scores_good)

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

    def reconstruction_error_histogram_per_scale(self, expect_fpr=0.3, max_step=5000):
        from sklearn.metrics import roc_auc_score, average_precision_score

        if self.load_model():
            print("Model Loaded.")
        else:
            print("None pretrained models.")
            return

        print("Calculating AUC, IOU, PRO metrics on testing data...")
        time_start = time.time()
        masks = []
        scores = []
        loss = 0
        recons_error_per_scale = [[],[],[],[]]
        n_examples = 0
        for i, (_, mask, name) in enumerate(self.test_data_loader):  # batch size is 1.
            # data
            mask = mask.squeeze().numpy()

            # anomaly score
            # anomaly_map = self.score(img).data.cpu().numpy()
            if "good" not in name[0]:
                continue
            n_examples += 1
            name = name[0].replace('data','dfr_output_scores').replace('\\test','')
            name = name[:-4] + ".npy"

            anomaly_map = np.load(name)

            anomap1 = np.mean(anomaly_map[:,:128], axis=1)
            anomap2 = np.mean(anomaly_map[:,128:384], axis=1)
            anomap3 = np.mean(anomaly_map[:,384:1408], axis=1)
            anomap4 = np.mean(anomaly_map[:,1408:3456], axis=1)
            
            recons_error_per_scale[0].append(anomap1.flatten().tolist())
            recons_error_per_scale[1].append(anomap2.flatten().tolist())
            recons_error_per_scale[2].append(anomap3.flatten().tolist())
            recons_error_per_scale[3].append(anomap4.flatten().tolist())


            masks.append(mask)

        # as array
        print("Average test loss : ", loss / len(self.test_data_loader))
        recons_error_per_scale = np.array(recons_error_per_scale)

        bins = np.linspace(0, np.max(recons_error_per_scale),255)

        hist_scale0, bins0 = np.histogram(recons_error_per_scale[0], bins = bins, density=True)
        hist_scale1, bins1 = np.histogram(recons_error_per_scale[1], bins = bins, density=True)
        hist_scale2, bins2 = np.histogram(recons_error_per_scale[2], bins = bins, density=True)
        hist_scale3, bins3 = np.histogram(recons_error_per_scale[3], bins = bins, density=True)

        plt.plot(bins0[:-1], hist_scale0, color="r", label = "Scale 1")
        plt.plot(bins1[:-1], hist_scale1, color="b", label = "Scale 2")
        plt.plot(bins2[:-1], hist_scale2, color="g", label = "Scale 3")
        plt.plot(bins3[:-1], hist_scale3, color="black", label = "Scale 4")
        plt.xlabel("Anomaly score (reconstruction error)")
        plt.legend()
        plt.show()
        
