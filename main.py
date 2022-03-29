import argparse
from anoseg_dfr import AnoSegDFR
from anoseg_unet import AnoSegUNET
import os

data_path = "C:\\Users\\Vincent\\MVA DELIRES\\Projet\\code\\data\\"
save_path = "C:\\Users\\Vincent\\MVA DELIRES\\Projet\\code\\cfg"

def config():
    parser = argparse.ArgumentParser(description="Settings of experience")

    # args to select argument
    parser.add_argument('--mode', type=str, choices=["train", "evaluation"],default="train", help="train or evaluation")
    parser.add_argument('--model_name', type=str, default="dfr", choices=["dfr", "riad", "unet"], help="specifed model name")
    parser.add_argument('--dfr_type', type=str, default="dfr", choices=["dfr", "dfr-s"], help="Ponderation of anoamly map according to features ? (yes:dfr-s, no:dfr)")
    parser.add_argument('--unet_size', type=int, default=512, help="if using model='unet', nb of features in latent space")
    parser.add_argument('--data_name', type=str, default="tile", choices=["tile","grid"],help="data name")

    # general
    parser.add_argument('--save_path', type=str, default=os.getcwd(), help="saving path")
    parser.add_argument('--img_size', type=int, nargs="+", default=(256, 256), help="image size (hxw)")
    parser.add_argument('--device', type=str, default="cpu", help="device for training and testing")
    parser.add_argument('--save_reconstruction', type=bool, default=False, help="save reconstruction images")

    # parameters for the dfr
    parser.add_argument('--backbone', type=str, default="vgg19", help="backbone net")

    cnn_layers = ('relu4_1', 'relu4_2', 'relu4_3', 'relu4_4')
    parser.add_argument('--cnn_layers', type=str, nargs="+", default=cnn_layers, help="cnn feature layers to use")
    parser.add_argument('--upsample', type=str, default="bilinear", help="operation for resizing cnn map")
    parser.add_argument('--is_agg', type=bool, default=True, help="if to aggregate the features")
    parser.add_argument('--featmap_size', type=int, nargs="+", default=(256, 256), help="feat map size (hxw)")
    parser.add_argument('--kernel_size', type=int, nargs="+", default=(4, 4), help="aggregation kernel (hxw)")
    parser.add_argument('--stride', type=int, nargs="+", default=(4, 4), help="stride of the kernel (hxw)")
    parser.add_argument('--dilation', type=int, default=1, help="dilation of the kernel")

    # training and testing
    # default values
    data_name = "bottle"
    train_data_path = "/cal/exterieurs/vfournet-21/DELIRES/data/" + data_name + "/train/good"
    test_data_path = "/cal/exterieurs/vfournet-21/DELIRES/data/" + data_name + "/test"

    parser.add_argument('--train_data_path', type=str, default=train_data_path, help="training data path")
    parser.add_argument('--test_data_path', type=str, default=test_data_path, help="testing data path")

    # CAE
    parser.add_argument('--latent_dim', type=int, default=None, help="latent dimension of CAE")
    parser.add_argument('--is_bn', type=bool, default=True, help="if using bn layer in CAE")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--epochs', type=int, default=150, help="epochs for training")    # default 700, for wine 150

    # segmentation evaluation
    parser.add_argument('--thred', type=float, default=0.5, help="threshold for segmentation")
    parser.add_argument('--except_fpr', type=float, default=0.005, help="fpr to estimate segmentation threshold")

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    #########################################
    #    On the whole data
    #########################################
    cfg = config()
    cfg.save_path = save_path

    # train or evaluation
    data_name = cfg.data_name
    cfg.train_data_path = data_path + data_name + "\\train\\good"
    cfg.test_data_path = data_path + data_name + "\\test"
    
        
    if cfg.model_name == "dfr":
        cfg.model_name = ""
        if data_name == "grid":
            cfg.latent_dim = 143
        elif data_name == "tile":
            cfg.latent_dim = 522
        else:
            cfg.latent_dim = None #Automatic selection of latent size for DFR
        
        # vgg features to use
        cfg.cnn_layers = ('relu1_1', 'relu1_2', 'relu2_1', 'relu2_2',
                    'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4',
                    'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4')

        model = AnoSegDFR(cfg)

    elif cfg.model_name == "riad" or cfg.model_name == "unet":
        model = AnoSegUNET(cfg)

    if cfg.mode == "train":
        model.train()
    elif cfg.mode == "evaluation":
        if cfg.model_name == "riad" or cfg.model_name == "unet" or cfg.dfr_type == "dfr":
            model.eval_segmentation()
        else:
            model.eval_segmentation_original()
