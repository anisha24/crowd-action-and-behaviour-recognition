import torch
import torch.nn as nn
import torch.nn.functional as F

class C3D(nn.Module):
    def __init__(self, num_classes=487):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)

        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        x = self.pool4(x)

        x = F.relu(self.conv5a(x))
        x = F.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = self.dropout(x)

        x = F.relu(self.fc7(x))
        x = self.dropout(x)

        x = self.fc8(x)

        return F.softmax(x, dim=1)


def get_model(input_shape=(3, 16, 112, 112), summary=False, num_classes=487):
    model = C3D(num_classes=num_classes)

    if summary:
        print(model)

    return model

if __name__ == '__main__':
    model = get_model(summary=True)




class Conv3DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation='relu'):
        super(Conv3DLayer, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, padding=(1, 1, 1))
        self.activation = nn.ReLU() if activation == 'relu' else None

    def forward(self, x):
        x = self.conv3d(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class MaxPool3DLayer(nn.Module):
    def __init__(self):
        super(MaxPool3DLayer, self).__init__()
        self.maxpool3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        return self.maxpool3d(x)

class FlattenLayer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class FC3DLayer(nn.Module):
    def __init__(self, in_features, out_features, activation='relu'):
        super(FC3DLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU() if activation == 'relu' else None

    def forward(self, x):
        x = self.fc(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class VGG16_3D(nn.Module):
    def __init__(self):
        super(VGG16_3D, self).__init__()

        self.conv1 = Conv3DLayer(3, 64, kernel_size=3)
        self.pool1 = MaxPool3DLayer()

        self.conv2 = Conv3DLayer(64, 128, kernel_size=3)
        self.pool2 = MaxPool3DLayer()

        self.conv3a = Conv3DLayer(128, 256, kernel_size=3)
        self.conv3b = Conv3DLayer(256, 256, kernel_size=3)
        self.pool3 = MaxPool3DLayer()

        self.conv4a = Conv3DLayer(256, 512, kernel_size=3)
        self.conv4b = Conv3DLayer(512, 512, kernel_size=3)
        self.pool4 = MaxPool3DLayer()

        self.conv5a = Conv3DLayer(512, 512, kernel_size=3)
        self.conv5b = Conv3DLayer(512, 512, kernel_size=3)
        self.zeropad = nn.ZeroPad3d(padding=(0, 0, 1, 1, 1, 1))
        self.pool5 = MaxPool3DLayer()

        self.flatten = FlattenLayer()

        self.fc6 = FC3DLayer(512 * 7 * 7 * 2, 4096)
        self.fc7 = FC3DLayer(4096, 4096)
        self.fc8 = FC3DLayer(4096, 487, activation='softmax')

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)

        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.zeropad(x)
        x = self.pool5(x)

        x = self.flatten(x)

        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)

        return x