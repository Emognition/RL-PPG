import torch
import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator_larfield
from trainer.trainer import Trainer, model_evaluate
from trainer.trainer_multimodal import MultimodalTrainer
from trainer.trainer_multimodal import model_evaluate as model_evaluate_multimodal
from models.TC import TC, TC_full_representation, TC_features
from utils import _calc_metrics, copy_Files
from models.model_multimodal import base_Model_ppg, base_Model_acc
# Args selections
start_time = datetime.now()


parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='LarField', type=str,
                    help='Dataset of choice: LarField, ProSi')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args = parser.parse_args()



device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'

training_mode = args.training_mode
run_description = args.run_description
if "pretrained_encoders" in experiment_description:
    pretrained_weights = True
else:
    pretrained_weights = False

    
logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
data_path = "/home/emognition/Desktop/LarField/"#"/home/emognition/Desktop/SSL-ARI/dataset/processed_dataset_60_sec" #f"../../LarFieldData/"
ppg_backbone=os.path.join(
        "/home/emognition/Desktop/multimodal-ssl/experiments_logs/ts_tcc_Larfield_initial_unimodal_ppg_fix/run_1/self_supervised_ppg_seed_123/saved_models")

acc_backbone = os.path.join(
    "/home/emognition/Desktop/multimodal-ssl/experiments_logs/ts_tcc_Larfield_initial_unimodal_acc_fix/run_1/self_supervised_acc_seed_123/saved_models")

#train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
train_dl, valid_dl, test_dl = data_generator_larfield(data_path, configs, training_mode, train_size_ablation=None)
logger.debug("Data loaded ...")

# Load Model
if "multimodal" in training_mode:
    model_ppg = base_Model_ppg(configs).to(device)
    model_acc = base_Model_acc(configs).to(device)
    if "full_representation" in training_mode:
        temporal_contr_model_ppg = TC_full_representation(configs, device).to(device)
        temporal_contr_model_acc = TC_full_representation(configs, device).to(device)
    if "feature" in training_mode:
        print("Feature learning started...")
        temporal_contr_model_ppg = TC_features(configs, device, num_features=1).to(device)
        temporal_contr_model_acc = TC_features(configs, device, num_features=1).to(device)
elif "acc" in experiment_description:
    model = base_Model_acc(configs).to(device)
    temporal_contr_model = TC(configs, device).to(device)
elif "ppg" in experiment_description:
    model = base_Model_ppg(configs).to(device)
    temporal_contr_model = TC(configs, device).to(device)

if training_mode == "fine_tune" or (pretrained_weights and "multimodal" in training_mode):
    # load saved model of this experiment
    load_from_ppg = os.path.join(ppg_backbone)
    chkpoint = torch.load(os.path.join(load_from_ppg, "ckp_best.pt"), map_location=device)
    # Load pretrained ppg model
    chkpoint = torch.load(os.path.join(load_from_ppg, "ckp_best.pt"), map_location=device)
    if "model_ppg_state_dict" not in chkpoint.keys():
        chkpoint["model_ppg_state_dict"] = chkpoint.pop("model_state_dict")
        torch.save(chkpoint, os.path.join(load_from_ppg, "ckp_best.pt"))

    pretrained_ppg_dict = chkpoint["model_ppg_state_dict"]
    model_ppg_dict = model_ppg.state_dict()
    del_list = ['logits']
    pretrained_ppg_dict_copy = pretrained_ppg_dict.copy()
    for i in pretrained_ppg_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_ppg_dict[i]
    model_ppg_dict.update(pretrained_ppg_dict)
    model_ppg.load_state_dict(model_ppg_dict)
    # Load pretrained acc model
    load_from_acc = os.path.join(acc_backbone)
    chkpoint = torch.load(os.path.join(load_from_acc, "ckp_best.pt"), map_location=device)
    if "model_acc_state_dict" not in chkpoint.keys():
        chkpoint["model_acc_state_dict"] = chkpoint.pop("model_state_dict")
        torch.save(chkpoint, os.path.join(load_from_acc, "ckp_best.pt"))
    pretrained_acc_dict = chkpoint["model_acc_state_dict"]
    model_acc_dict = model_acc.state_dict()
    pretrained_acc_dict_copy = pretrained_acc_dict.copy()
    for i in pretrained_acc_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_acc_dict[i]
    model_acc_dict.update(pretrained_acc_dict)
    model_acc.load_state_dict(model_acc_dict)
if training_mode == "fine_tune" and "multimodal" not in training_mode:
    if "ppg" in training_mode:
        # load saved model of this experiment
        load_from_ppg = os.path.join(ppg_backbone)
        chkpoint = torch.load(os.path.join(load_from_ppg, "ckp_best.pt"), map_location=device)
        # Load pretrained ppg model
        pretrained_ppg_dict = chkpoint["model_ppg_state_dict"]
        model_ppg_dict = model.state_dict()
        del_list = ['logits']
        pretrained_ppg_dict_copy = pretrained_ppg_dict.copy()
        for i in pretrained_ppg_dict_copy.keys():
            for j in del_list:
                if j in i:
                    del pretrained_ppg_dict[i]
        model_ppg_dict.update(pretrained_ppg_dict)
        model.load_state_dict(model_ppg_dict)
    elif "acc" in training_mode:
        # Load pretrained acc model
        load_from_acc = os.path.join(acc_backbone)
        chkpoint = torch.load(os.path.join(load_from_acc, "ckp_best.pt"), map_location=device)
        pretrained_acc_dict = chkpoint["model_acc_state_dict"]
        model_acc_dict = model.state_dict()
        pretrained_acc_dict_copy = pretrained_acc_dict.copy()
        for i in pretrained_acc_dict_copy.keys():
            for j in del_list:
                if j in i:
                    del pretrained_acc_dict[i]
        model_acc_dict.update(pretrained_acc_dict)
        model.load_state_dict(model_acc_dict)
if training_mode == "train_linear" or "tl" in training_mode:
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # delete these parameters (Ex: the linear layer at the end)
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

if training_mode == "random_init":
    model_dict = model.state_dict()

    # delete all the parameters except for logits
    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.


if "multimodal" in training_mode:
    model_ppg_optimizer = torch.optim.Adam(model_ppg.parameters(), lr=configs.ppg_lr,
                                           betas=(configs.ppg_beta1, configs.ppg_beta2), weight_decay=3e-4)
    model_acc_optimizer = torch.optim.Adam(model_acc.parameters(), lr=configs.acc_lr,
                                           betas=(configs.acc_beta1, configs.acc_beta2), weight_decay=3e-4)
    temporal_contr_optimizer_ppg = torch.optim.Adam(temporal_contr_model_ppg.parameters(), lr=configs.lr,
                                                    betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    temporal_contr_optimizer_acc = torch.optim.Adam(temporal_contr_model_acc.parameters(), lr=configs.lr,
                                                    betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

else:
    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

if "self_supervised" in training_mode:  # to do it only once
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

if "multimodal" in training_mode:
    MultimodalTrainer(model_ppg, model_acc, temporal_contr_model_ppg, temporal_contr_model_acc, model_ppg_optimizer, model_acc_optimizer,
                      temporal_contr_optimizer_ppg, temporal_contr_optimizer_ppg, train_dl, valid_dl, test_dl, device, logger, configs,
                      experiment_log_dir, training_mode)
else:
    # Trainer
    Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode)

if "self_supervised" not in training_mode:
    # Testing
    if "multimodal" in training_mode:
        outs = model_evaluate_multimodal(model_ppg, model_acc, temporal_contr_model_ppg, temporal_contr_model_acc, test_dl, device, training_mode)
    else:
        outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
    total_loss, total_acc, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

logger.debug(f"Training time is : {datetime.now()-start_time}")
