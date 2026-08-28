import importlib
import sys
import torch

import pytorch_lightning as pl
from torch.nn import ModuleDict

sys.path.append('./SSL-model')


def load_encoder(model_name: str, run_name: str, is_multimodal: bool, modality: str,
                 backbone_fine_tuning_strategy: str = "freeze",
                 encoder_from_scratch: bool = False):
    if isinstance(modality, list):
        raise TypeError("a modality must be a str (single value instead of list)")

    LarField_Configs = importlib.import_module(
        f'experiments_logs.{model_name}.{run_name}.model_files.LarField_Configs')
    Configs = LarField_Configs.Config

    model_multimodal = importlib.import_module(
        f'experiments_logs.{model_name}.{run_name}.model_files.model_multimodal')
    if modality.lower() == 'ecg':
        base_Model = model_multimodal.base_Model_ecg
    elif modality.lower() == 'acc':
        base_Model = model_multimodal.base_Model_acc
    elif modality.lower() == 'ppg':
        base_Model = model_multimodal.base_Model_ppg
    else:
        raise f"No model for {modality} in {model_name}"

    if "feature_learning" in model_name:
        multimodal_string = "_multimodal_feature_learning"
    elif is_multimodal:
        multimodal_string = "_multimodal"
    else:
        multimodal_string = f'_{modality.lower()}'


    load_from = (f"./SSL-model/experiments_logs/"
                 f"{model_name}/{run_name}/self_supervised{multimodal_string}_seed_123/"
                 f"saved_models/ckp_best.pt")

    if 'feature_learning_1_1' in model_name:
        load_from = (f"./SSL-model/experiments_logs/"
                     f"{model_name}/{run_name}/self_supervised_multimodal_feature_learning_seed_123/"
                     f"saved_models/ckp_best.pt")


    device = torch.device("cuda")
    larfield_configs = Configs()

    # Load Model
    model = base_Model(larfield_configs).to(device)

    if not encoder_from_scratch:
        try:
            chkpoint = torch.load(load_from, map_location=device)
        except:
            try:
                load_from = (f"./SSL-model/experiments_logs/"
                             f"{model_name}/{run_name}/self_supervised_personal_{model_name.split('_')[-1]}_seed_123/"
                             f"saved_models/ckp_best.pt")
                chkpoint = torch.load(load_from, map_location=device)

            except:
                load_from = (f"./SSL-model/experiments_logs/"
                             f"{model_name}/{run_name}/self_supervised_acc_seed_123/"
                             f"saved_models/ckp_best.pt")
                chkpoint = torch.load(load_from, map_location=device)




        # load weights to encoder
        try:
            pretrained_dict = chkpoint[f"model_{modality.lower()}_state_dict"]
        except:
            pretrained_dict = chkpoint[f"model_state_dict"]

        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    if backbone_fine_tuning_strategy == "freeze":
        model.eval()
    elif backbone_fine_tuning_strategy == "finetune":
        model.train()
    return model, larfield_configs.final_out_channels


def load_ts_tcc_encoders(modalities_model_names: {str: str}, run_name: str, is_multimodal: bool,
                         backbone_fine_tuning_strategy: str = "freeze", encoder_from_scratch: bool = False):
    pl.seed_everything(123, workers=True)

    if "finetune" in backbone_fine_tuning_strategy:
        encoders, representation_dimensionality = ModuleDict(), None
    else:
        encoders, representation_dimensionality = {}, None
    for modality, model_name in modalities_model_names.items():
        model, representation_dimensionality = load_encoder(model_name, run_name, is_multimodal,
                                                            modality=modality.lower(),
                                                            backbone_fine_tuning_strategy=backbone_fine_tuning_strategy,
                                                            encoder_from_scratch=encoder_from_scratch)
        encoders[modality.upper()] = model

    return encoders, representation_dimensionality
