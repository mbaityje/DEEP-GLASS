from models.generic_models import conv22tanh,conv22relu,loadNet,convAlexrelu,AlexNet

def convTest(pretrained_path=False, **kwargs):
    """Constructs a very simple convNet
    Args:
        pretrained_path (bool): If True, returns the  pretrained model on the path
    """
    model = conv22relu(layers=[3, 2, 4, 32,100],im_dim=32)
    if pretrained_path:
        return loadNet(pretrained_path,model)
    return model

def conv1020relu(pretrained_path=False, **kwargs):
    """Constructs a very simple convNet
    Args:
        pretrained_path (bool): If True, returns the  pretrained model on the path
    """
    model = conv22relu(layers=[3, 10, 32, 400,100],im_dim=32)
    if pretrained_path:
        return loadNet(pretrained_path,model)
    return model

def conv1020tanh(pretrained_path=False, **kwargs):
    """Constructs a very simple convNet
    Args:
        pretrained_path (bool): If True, returns the  pretrained model on the path
    """
    model = conv22tanh(layers=[3, 10, 32, 400,100],im_dim=32)
    if pretrained_path:
        return loadNet(pretrained_path,model)
    return model

def alex6464(pretrained_path=False, **kwargs):
    """alex like convnet
    Args:
        pretrained_path (bool): If True, returns the  pretrained model on the path
    """
    model = AlexNet(layers=[3,64,128,512,128,100])
    if pretrained_path:
        return loadNet(pretrained_path,model)
    return model

