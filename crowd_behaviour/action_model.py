import torch
import torch.nn as nn

class AppearanceBranch(nn.Module):
    def __init__(self):
        super(AppearanceBranch, self).__init__()
        
        self.branch_appearance = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5),
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5),
            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten()
        )
    
    def forward(self, x):
        out_appearance = self.branch_appearance(x)
        return out_appearance
    
class MotionBranch(nn.Module):
    def __init__(self):
        super(MotionBranch, self).__init__()
        
        self.branch_motion = nn.Sequential(
            nn.Conv2d(20, 96, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5),
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5),
            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten()
        )
    
    def forward(self, x):
        out_motion = self.branch_motion(x)
        return out_motion
    
class DeeplyLearnedAttributes(nn.Module):
    def __init__(self):
        super(DeeplyLearnedAttributes, self).__init__()
        
        self.appearance_branch = AppearanceBranch()
        self.motion_branch = MotionBranch()
        self.concat = nn.Linear(4096*2, 51)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_appearance, x_motion):
        out_appearance = self.appearance_branch(x_appearance)
        out_motion = self.motion_branch(x_motion)
        out = torch.cat((out_appearance, out_motion), dim=1)
        out = self.concat(out)
        out = self.sigmoid(out)
        return out

model = DeeplyLearnedAttributes()

print(model)