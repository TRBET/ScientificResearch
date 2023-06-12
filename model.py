import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 8)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 8)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8), padding=(0, 3)),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 8)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 8)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8), padding=(0, 2)),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 8)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 8)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8), padding=(0, 1)),
            nn.Dropout(0.25)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 6, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 6)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    from dataset import train_loader
    # import torch
    # net = Net()
    # for i, j in train_loader:
    #     pred = net(i)
    #     print(pred)
    #     print(j.data)
    #     prd_lab = torch.argmax(pred, 1)
    #     print(torch.sum(prd_lab == j.data))
    #     break
    # print(len(train_loader))
    net = Net()
    for i, j in train_loader:
        pred = net(i)
        print(pred.shape)
        print(pred)
        break
