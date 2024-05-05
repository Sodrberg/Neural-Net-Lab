import torch
from torchvision import models
from torch import nn

classes = ['articleType', 'Shirts', 'Pants', 'Watches', 'T-shirts', 'Accessories', 'Shoes', 'Belts', 'Slippers', 'Bags', 'Sweaters',
           'Outerwear', 'Shorts', 'Dresses', 'Robe', 'Skirts', 'Hats', 'One-piece', 'Swimwear', 'Ties', 'Headband', 'Gloves', 'Umbrellas', 'Suits']

model = models.resnet50() 

num_classes = 24 
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load('./clothing-model.pth',
                      map_location=torch.device('cpu')))

model.eval()

def predict(image_path):
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0) 

    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)

    return predicted.item()

if __name__ == "__main__":
    image_path = '../../Large-mixed-clothing-dataset/images/1533.jpg' #testa olika bilder här
    prediction = predict(image_path)
    for i in classes:
        if prediction == classes.index(i):
            print(f'Predicted class: {i}')
            break
