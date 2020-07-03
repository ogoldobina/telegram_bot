from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
from tqdm import tqdm
from datetime import datetime


class MessageRW:
    def __init__(self, bot, message, show_last_update=False):
        self.bot = bot
        self.chat_id = message.chat.id
        self.message_id = message.message_id
        self.text = message.text
        self.show_last_update = show_last_update
        self.prev_msg = self.text

    def write(self, s):
        new_text = s.strip().replace('\r', '')
        if len(new_text) != 0:
            self.text = new_text

    def flush(self):
        pass

    async def send(self):
        msg = self.text + ('\nLast update: {}'.format(datetime.now()) if self.show_last_update else '')
        if self.prev_msg != msg:
            await self.bot.edit_message_text(msg, self.chat_id, self.message_id)
            self.prev_msg = msg

def load_img(path, size=128):
    image = Image.open(path)
    loader = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = loader(image).unsqueeze(0)
    return image

def imshow(tensor, title=None):
    unloader = transforms.Compose([
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Lambda(lambda x: x.clamp(0, 1)),
        transforms.ToPILImage() 
    ])
    
    image = tensor.cpu().clone()   
    image = image.squeeze(0)    
    image = unloader(image)
    
    plt.ion()
    plt.figure()
    plt.imshow(image)
    if title is not None:
        plt.title(title)
        
        
def gram_matrix(input):
    b, c, h, w = input.size()  
    features = input.view(c, h * w) 

    G = torch.mm(features, features.t())
    return G.div(h * w) 

class Loss:
    def __init__(self, model, content_img, style_img):
        self.model = model
        self.content_layers = ['19']
        self.style_layers = ['0', '5', '10', '17', '24']
        
        x = content_img.detach()
        self.content_features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.content_layers:
                self.content_features[name] = x.detach()
        
        x = style_img.detach()
        self.style_features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.style_layers:
                self.style_features[name] = gram_matrix(x).detach()
        
    def get_losses(self, input_img):
        x = input_img
        content_loss = 0
        style_loss = 0
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.content_layers:
                content_loss += F.mse_loss(x, self.content_features[name])
            if name in self.style_layers:
                style_loss +=  1e3 / layer.out_channels ** 2 * F.mse_loss(gram_matrix(x), self.style_features[name])
        return content_loss, style_loss
    
class ST_model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        cnn = models.vgg16(pretrained=True).features.to(self.device).eval()        
        self.model = nn.Sequential()
        for name, layer in cnn._modules.items():
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            for p in layer.parameters():
                p.requires_grad_(False)

            self.model.add_module(name, layer)
            
    async def run_style_transfer(self, content_img, style_img, progress_msg:MessageRW, num_steps=500,
                        style_weight=1e4, content_weight=1):

        content_img = content_img.to(self.device)
        style_img = style_img.to(self.device)

        input_img = content_img.clone().requires_grad_(True)
        input_img = input_img.to(self.device)

        optimizer = optim.LBFGS([input_img])
        L = Loss(self.model, content_img, style_img)

        def closure():            
            optimizer.zero_grad()
            content_loss, style_loss = L.get_losses(input_img)

            loss = style_weight * style_loss + content_weight * content_loss
            loss.backward()

            nonlocal run, rg
            run += 1
            try:
                rg.update()
            except StopIteration:
                pass
            return loss

        
        run = 0
        rg = tqdm(iterable=range(1,num_steps+1), file=progress_msg)
        await progress_msg.send()
        while run < num_steps:
            optimizer.step(closure)
            await progress_msg.send()

        unloader = transforms.Compose([
            transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                 std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Lambda(lambda x: x.clamp(0, 1)),
            transforms.ToPILImage() 
        ])
        
        input_img = unloader(input_img.squeeze(0))    
        return input_img 