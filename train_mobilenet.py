import transforms








data_transform = {
    'train': transforms.Compose([transforms.ToTensor(),
                                 transforms.RandomHorizonalFlip(0.5)]),
    'val': transforms.Compose([transforms.ToTensor()])
}