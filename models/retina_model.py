import torchvision
import torch 

class RetinaRehead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights='DEFAULT')
        self.detector = torch.nn.Conv2d(256, 10, kernel_size=1)
    
    def forward(self, input):
        res = self.model.backbone.forward(input)
        res = res['0']
        res = self.detector.forward(res)
        return res