from torch import nn, squeeze, mean

size_factor = 1
class base_Model_ppg(nn.Module):
    def __init__(self, configs):
        super(base_Model_ppg, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.ecg_input_channels, 32*size_factor, kernel_size=configs.ecg_kernel_size,
                      stride=configs.ecg_stride, bias=False, padding=(configs.ecg_kernel_size//2)),
            nn.BatchNorm1d(32*size_factor),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32*size_factor, 64*size_factor, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64*size_factor),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        # Adding more layers than in original TS-TCC
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64*size_factor, 128*size_factor, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128*size_factor),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(128*size_factor, 256*size_factor, kernel_size= configs.ecg_kernel_size//2, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(256*size_factor),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        """
        self.conv_block4_2 = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )   
        """
        # end of more layers

        self.conv_block5 = nn.Sequential(
            nn.Conv1d(256*size_factor, configs.ecg_final_out_channels, kernel_size=4, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.ecg_final_out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )


        self.logits = nn.Linear(configs.ecg_final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in).to(x_in.device)

        x = self.conv_block2(x)

        x = self.conv_block3(x)

        x = self.conv_block4(x)

        #x = self.conv_block4_2(x)

        x = self.conv_block5(x)

        x_mean = mean(x, -1)
        logits = self.logits(x_mean)
        return logits, x


class base_Model_acc(nn.Module):
    def __init__(self, configs):
        super(base_Model_acc, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.acc_input_channels, 32*size_factor, kernel_size=configs.acc_kernel_size,
                      stride=configs.acc_stride, bias=False, padding=(configs.acc_kernel_size // 2)),
            nn.BatchNorm1d(32*size_factor),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32*size_factor, 64*size_factor, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64*size_factor),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        # Adding more layers than in original TS_TCC
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64*size_factor, 128*size_factor, kernel_size=4, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(128*size_factor),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(128*size_factor, configs.acc_final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.acc_final_out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        # end of more layers
        """
        self.conv_block5 = nn.Sequential(
            nn.Conv1d(256, configs.acc_final_out_channels, kernel_size=4, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(configs.acc_final_out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
        )
        """
        self.logits = nn.Linear(configs.acc_final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in).to(x_in.device)

        x = self.conv_block2(x)

        x = self.conv_block3(x)

        x = self.conv_block4(x)

        # x = self.conv_block5(x)

        x_mean = mean(x, -1)
        logits = self.logits(x_mean)

        return logits, x
