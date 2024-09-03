from torchvision import transforms



def get_transforms_train():
# Define the dataset object
    transform = transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()) ,
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Normalize(mean=[(0.485+0.456+0.406)/3], std=[(0.229+ 0.224+ 0.225)/3]),
    ])

    return transform




def get_transforms_val():
    transform = transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()) ,
        transforms.Resize((224, 224)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[(0.485+0.456+0.406)/3], std=[(0.229+ 0.224+ 0.225)/3]),

        
    ])


    return transform