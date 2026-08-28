class Config(object):
    def __init__(self):
        # ECG model configs
        self.ecg_input_channels = 1
        self.ecg_kernel_size = 8
        self.ecg_stride = 1
        self.ecg_final_out_channels = 128
        self.ecg_features_len = 34

        # TC
        self.final_out_channels = 128


        # ACC model configs
        self.acc_axis=[0,1,2] # 0 - x, 1 - y, 2 - z, 3 - magnitude
        self.acc_input_channels = len(self.acc_axis)
        self.acc_kernel_size =8
        self.acc_stride = 1
        self.acc_final_out_channels = 128
        self.acc_features_len = 18


        self.num_classes = 2
        self.dropout = 0.35

        # training configs
        self.num_epoch = 40

        # ECG optimizers parameters
        self.ecg_beta1 = 0.9
        self.ecg_beta2 = 0.99
        self.ecg_lr = 3e-2#3e-1

        # ACC optimizers parameters
        self.acc_beta1 = 0.9
        self.acc_beta2 = 0.99
        self.acc_lr = 3e-2#3e-1

        # TC optimizers parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr =3e-2#3e-1

        # data parameters
        self.drop_last = True
        self.batch_size = 128

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

        self.ecg_projection_dim = 128
        self.acc_projection_dim = 128

        # TODO: add outputs' dimensions for other downstream tasks

        self.prosi_multiclass_output_dim = 5
        self.prosi_binary_output_dim = 2


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.01
        self.jitter_ratio = 0.001
        self.max_seg = 5


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 128
        self.timesteps = 10
