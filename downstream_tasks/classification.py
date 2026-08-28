import argparse
import gc
from datetime import datetime
from random import seed

import torch
import yaml
import pytorch_lightning as pl
from copy import deepcopy
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from downstream_tasks.models.ClassificationHeadAttention import ClassificationHeadAttention
from downstream_tasks.models.ClassificationHeadLinear import ClassificationHeadLinear
from downstream_tasks.models.ClassificationHeadMLP import ClassificationHeadMLP
from downstream_tasks.encoders_utils import load_ts_tcc_encoders
from larfield_emo.dataloader import LarFieldEmo
from larfield_physical_activity.dataloader import LarFieldActivity
from prosi.dataloader import ProSi


# config overwrites values from config files
def run_experiment(config: {}) -> None:
    pl.seed_everything(123, workers=True)
    seed(123)

    config.modalities = list(config.modalities_model_names.keys())
    config.is_multimodal = True if ('multimodal'
                                    in list(config.modalities_model_names.values())[0]) else False

    if config.dataset == "LarFieldEmo" or config.dataset.lower() == "larfieldemo":
        dataset = LarFieldEmo(dataset_path=config.dataset_dir)
    elif config.dataset == "LarFieldActivity" or config.dataset.lower() == "larfieldactivity":
        dataset = LarFieldActivity(dataset_path=config.dataset_dir)
    elif config.dataset == "ProSi" or config.dataset.lower() == "prosi":
        dataset = ProSi(dataset_path=config.dataset_dir, modalities=config.modalities)

    p_list = dataset.get_participant_list()

    # available tasks are defined in the set_task()
    dataset.set_task(config.task)

    if config.task_type_regression:
        config.output_dim = 1
    else:
        config.output_dim = dataset.get_num_classes()

    # the int value is a number of folds to prepare - not a number of subjects to exclude for testing
    fold_type = config.fold_type
    if fold_type.upper() == "LOSO":
        num_folds = len(p_list)
    elif fold_type.upper() == "LKSO":
        num_folds = config.num_folds
    else:
        print("Specify fold type in config file: LOSO or LKSO")
        num_folds = 0

    # GroupKFold - could be used
    kf = KFold(n_splits=num_folds, shuffle=False)
    for i, (train, test) in enumerate(kf.split(p_list)):
        test_participants = [p_list[x] for x in test]
        train_idxes, test_idxes, val_idxes = dataset.get_participants_train_test_split(
            test_participants, validation_size=config.validation_size, train_size_ablation=config.train_size_ablation,
            LOSO=config.LOSO, across_time=config.across_time, cold_start=config.cold_start)
        print()
        print(f"Fold {i}:")
        print(f"  Testing participants:  {test_participants}")

        test_dataset = Subset(dataset, test_idxes)
        train_dataset = Subset(dataset, train_idxes)
        val_dataset = Subset(dataset, val_idxes)

        test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=1)
        train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1)

        all_labels = train_dataset.dataset.labels[config.task].tolist()
        train_labels = [int(all_labels[idx]) for idx in train_dataset.indices]
        num_classes = len(set(all_labels))
        loss_weights = []
        for j in range(num_classes):
            samples_label_j = [x for x in train_labels if x == j]
            loss_weights.append(len(train_labels) / (len(samples_label_j) * num_classes))

        if config.use_backbone:
            encoders, representation_dimensionality = load_ts_tcc_encoders(
                config.modalities_model_names,
                config.run_name,
                config.is_multimodal,
                config.backbone_fine_tuning_strategy,
                config.encoder_from_scratch
            )
            config.projection_dim = representation_dimensionality
        else:
            encoders = None
            config.projection_dim = dataset.get_sample_dim()

        if config.classificationHead.lower() == 'classificationheadlinear':
            classificationHead = ClassificationHeadLinear(args=config, encoders=encoders)
        elif config.classificationHead.lower() == 'classificationheadattention':
            classificationHead = ClassificationHeadAttention(args=config, encoders=encoders)
        elif config.classificationHead.lower() == 'classificationheadmlp':
            classificationHead = ClassificationHeadMLP(args=config, encoders=encoders, loss_weights=loss_weights)
        else:
            raise NotImplementedError(f"Not handled classificationHead: {config.classificationHead}")

        encoder_from_scratch_label = "_not_loaded_checkpoint" if config.encoder_from_scratch else ""
        fusion_strategy_label = "fusion_mean" if config.fusion_strategy else "fusion_concat"
        backbone_fine_tuning_strategy_label = "_finetune" if config.backbone_fine_tuning_strategy == "finetune" else ""
        across_time_label = "_across_time" if config.across_time == True else ""
        cold_start_label = "_cold_start" if config.cold_start == True else ""
        LOSO_label = "_personal" if config.LOSO == False else ""
        train_size_ablation_label = f"_train_size_ablation_{str(config.train_size_ablation)}" if config.train_size_ablation else ""

        logger = TensorBoardLogger(
            save_dir=config.log_path,
            name=f"{config.experiment_name}_{encoder_from_scratch_label}_{fusion_strategy_label}{backbone_fine_tuning_strategy_label}{across_time_label}{cold_start_label}{LOSO_label}"
                 f"_{config.task}_{str(dataset.get_num_classes()) + 'classes' if not config.task_type_regression else 'regression'}"
                 f"_lr_{str(config.learning_rate)}"
                 f"{train_size_ablation_label}"
                 f"__{'_'.join(config.modalities)}",
            version=f"{config.classificationHead}_{fold_type}"
                    f"__{'_'.join(test_participants)}",
        )

        model_ckpt_callback = ModelCheckpoint(
            monitor="Valid/loss" if config.task_type_regression else "Valid/f1",
            mode="min" if config.task_type_regression else "max",
            save_top_k=1,
            dirpath=f"{config.ckpt_path}/{logger.name}/{logger.version}"
        )

        trainer = Trainer(
            logger=logger, max_epochs=config.max_epochs,
            deterministic=True, accelerator="gpu",
            callbacks=[model_ckpt_callback],
        )

        trainer.fit(
            model=classificationHead, train_dataloaders=train_loader, val_dataloaders=val_loader,
        )
        trainer.test(
            model=classificationHead, dataloaders=test_loader, ckpt_path='best',
        )

        del logger
        del trainer
        del classificationHead
        del encoders

        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    configs = []
    dataset = 'larfieldactivity'  # 'larfieldemo'#'prosi'
    tasks = ['binary']  # ['intense_emotion']
    for train_size_ablation in [100]:
        for learning_rate in [0.1, 0.05, 0.005, 0.0005]:
            for model_name in ["larfield_ppg_10_sec_full_3e_3_26"]:
                for modalities in [["PPG"]]:
                    modalities_model_names = {modality: model_name for modality in modalities}
                    if ('unimodal' in model_name and [True for modality, model in modalities_model_names.items()
                                                      if modality.lower() not in model.lower()]):
                        continue

                    for task in tasks:
                        configs.append({
                            'dataset': dataset,
                            'experiment_name': model_name,
                            'modalities_model_names': modalities_model_names,
                            'encoder_from_scratch': False, 'fusion_strategy': 'mean',
                            'task': task,
                            'train_size_ablation': train_size_ablation,
                            'backbone_fine_tuning_strategy': "freeze",  # "finetune",
                            'cold_start': False,
                            'across_time': False,
                            'LOSO': True,
                            'learning_rate': learning_rate,
                            'classificationHead': 'ClassificationHeadMLP',
                            'fold_type': 'LOSO',
                            'max_epochs': 200
                        })
            """
            for task in tasks:
                configs.append({
                    'dataset': dataset,
                    'config_model_name': "ts_tcc_Larfield_initial_unimodal_fusion",
                    'modalities_model_names': {
                        "ECG": "ts_tcc_Larfield_initial_unimodal_ecg_fix",
                        "ACC": "ts_tcc_Larfield_initial_unimodal_acc_fix",
                    },
                    'encoder_from_scratch': False, 'fusion_strategy': 'mean',
                    'task': task,
                    'learning_rate': learning_rate,
                    'train_size_ablation': train_size_ablation,
                    'classificationHead': 'ClassificationHeadMLP',
                    'fold_type': 'LOSO',
                'max_epochs': 200
            })
            """
        counter = 1

        # config parser
        parser = argparse.ArgumentParser(description="Supervised learning task")
        parser.add_argument("--config", default="", type=str)
        args = parser.parse_args()

        # TODO: set the paths in the config file
        config_file = f"/home/emognition/Desktop/ppg-ssl/downstream_tasks/config/config_{configs[0]['dataset']}.yaml"
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            parser.add_argument(f"--{key}", default=value, type=type(value))
        file_config = parser.parse_args()

        for config in configs:
            start_time = datetime.now()
            updated_config = deepcopy(file_config)
            for key, value in config.items():
                updated_config.__setattr__(key, value)

            print()
            print(f"Running experiment {counter}/{len(configs)}")
            print(f"    config: {config}")
            print()
            run_experiment(updated_config)
            print()
            print(f"Finished experiment {counter}/{len(configs)}")
            print(f"Start time: {start_time}; 1 experiment time: {datetime.now() - start_time}")
            print()
            counter += 1
