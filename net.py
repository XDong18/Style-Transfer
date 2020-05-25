import torchvision.models as models
from torchvision import transforms
import torch


class nn_Normalization(torch.nn.Module):
    """Normalize input image."""

    def __init__(self, mean, std):
        super(nn_Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).cuda()
        self.std = torch.tensor(std).view(-1, 1, 1).cuda()

    def forward(self, img):
        """Forward pass."""
        return (img - self.mean) / self.std


class vgg19_features(object):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, vgg_net, content_name_list, style_name_list, device):
        self.device = device
        self.features = list(vgg_net.children())[0].to(self.device)
        self.features.eval()
        self.content_name_list = content_name_list
        self.style_name_list = style_name_list
        self.normalize = nn_Normalization(self.MEAN, self.STD)
    
    def extract_features(self, img):
        output = img.clone()
        output = self.normalize(output)
        content_output_list = []
        style_output_list = []
        level = 0
        stage = 1
        for layer in self.features.children():
            if isinstance(layer, torch.nn.Conv2d):
                level += 1
                name = 'conv{0}_{1}'.format(stage, level)
            elif isinstance(layer, torch.nn.MaxPool2d):
                level = 0
                stage += 1
                name = None
            else:
                name = None

            output = layer(output)
            if name in self.content_name_list:
                content_output_list.append(output)
                # print('content', output.size())

            if name in self.style_name_list:
                style_output_list.append(output)
                # print('style', output.size())
        
        return content_output_list, style_output_list
    
if __name__ == "__main__":
    features = list(vgg19.children())[0]
    features.eval()
    for layer in features.children():
        if isinstance(layer, torch.nn.Conv2d):
            print(layer)                                 
