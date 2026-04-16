import torchvision.models as models

# Define model configurations
MODEL_CONFIGS = {
    'vgg-13': {
        'factory': lambda: models.vgg13(weights=models.VGG13_Weights.DEFAULT),
        'size': 224,
        'source': 'torchvision'
    },
    'vgg-16': {
        'factory': lambda: models.vgg16(weights=models.VGG16_Weights.DEFAULT),
        'size': 224,
        'source': 'torchvision'
    },
    'vgg-19': {
        'factory': lambda: models.vgg19(weights=models.VGG19_Weights.DEFAULT),
        'size': 224,
        'source': 'torchvision'
    },
    'efficientnet-b0': {
        'factory': lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
        'size': 224,
        'source': 'torchvision'
    },
    'efficientnet-b1': {
        'factory': lambda: models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT),
        'size': 240,
        'source': 'torchvision'
    },
    'efficientnet-b2': {
        'factory': lambda: models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT),
        'size': 260,
        'source': 'torchvision'
    },
    'efficientnet-b3': {
        'factory': lambda: models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT),
        'size': 300,
        'source': 'torchvision'
    },
    'efficientnet-b4': {
        'factory': lambda: models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT),
        'size': 380,
        'source': 'torchvision'
    },
    'efficientnet-b5': {
        'factory': lambda: models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT),
        'size': 456,
        'source': 'torchvision'
    },
    'efficientnet-b6': {
        'factory': lambda: models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.DEFAULT),
        'size': 528,
        'source': 'torchvision'
    },
    'efficientnet-b7': {
        'factory': lambda: models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT),
        'size': 600,
        'source': 'torchvision'
    },
    'resnet50': {
        'factory': lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        'size': 224,
        'source': 'torchvision'
    },
    'resnet152': {
        'factory': lambda: models.resnet152(weights=models.ResNet152_Weights.DEFAULT),
        'size': 224,
        'source': 'torchvision'
    },
    'inception-v3': {
        'factory': lambda: models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=True),
        'size': 299,
        'source': 'torchvision'
    }
}