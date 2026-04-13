import torch
import torchvision
import os
from ghn3 import from_pretrained

ghn = from_pretrained('ghn3tm8.pt')
model = torchvision.models.resnet50()
model = ghn(model)
print('GHN-3 predicted weights for ResNet-50 successfully')

os.makedirs('./checkpoints', exist_ok=True)

checkpoint = {
    'state_dict': model.state_dict(),
    'epoch': 0,
    'step': 0
}
torch.save(checkpoint, './checkpoints/resnet50_ghn3_predicted.pt')
print('Checkpoint saved to ./checkpoints/resnet50_ghn3_predicted.pt')