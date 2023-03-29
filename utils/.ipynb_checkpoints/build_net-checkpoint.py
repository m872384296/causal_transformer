from models.swin_transformer_v2 import SwinTransformerV2
from models.transformer_decoder import decoder
from models.split_parameter import split_param
from torch.nn import ModuleList

def build_swinv2(config, num_classes):
    net = SwinTransformerV2(img_size=config['img_size'],
                            num_classes=num_classes,
                            embed_dim=192,
                            depths=[2, 2, 18, 2],
                            num_heads=[6, 12, 24, 48],
                            window_size=24,
                            drop_path_rate=0.2,
                            pretrained_window_sizes=[12, 12, 12, 6])
    return ModuleList([net])

def build_all_net(config, init_trainloader):
    cls_net = build_swinv2(config, init_trainloader.num_classes)[0]
    spl_net = decoder(init_trainloader.dim_conf, init_trainloader.num_classes)
    spl_param = split_param(init_trainloader.num_train, config['n_env'])
    net = ModuleList([cls_net, spl_net, spl_param])
    return net
    