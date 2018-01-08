from models.generic_models import conv22tanh,conv22relu,loadNet,singleHiddenFullyConnected

def bruna10(pretrained_path=False, hidden_size=10,**kwargs):
    """Constructs a very simple convNet
    Args:
        pretrained_path (bool): If True, returns the pretrained model on the path
    """
    model = singleHiddenFullyConnected(layers=[784,hidden_size,10])
    if pretrained_path:
        return loadNet(pretrained_path,model)
    return model

def convTest(pretrained_path=False, **kwargs):
    """Constructs a very simple convNet
    Args:
        pretrained_path (bool): If True, returns the  pretrained model on the path
    """
    model = conv22relu(layers=[1, 2, 4, 32,10])
    if pretrained_path:
        return loadNet(pretrained_path,model)
    return model

def conv1020relu(pretrained_path=False, **kwargs):
    """Constructs a very simple convNet
    Args:
        pretrained_path (bool): If True, returns the  pretrained model on the path
    """
    model = conv22relu(layers=[1, 10, 20, 80,10])
    if pretrained_path:
        return loadNet(pretrained_path,model)
    return model

def conv1020tanh(pretrained_path=False, **kwargs):
    """Constructs a very simple convNet
    Args:
        pretrained_path (bool): If True, returns the  pretrained model on the path
    """
    model = conv22tanh(layers=[1, 10, 20, 80,10])
    if pretrained_path:
        return loadNet(pretrained_path,model)
    return model


