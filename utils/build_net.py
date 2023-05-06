from models.swin_transformer_v2 import SwinTransformerV2
from models.transformer_decoder import decoder
from torch import nn

def build_swinv2(config, num_classes):
    net = SwinTransformerV2(img_size=config['img_size'],
                            num_classes=num_classes,
                            embed_dim=192,
                            depths=[2, 2, 18, 2],
                            num_heads=[6, 12, 24, 48],
                            window_size=24,
                            drop_path_rate=0.2,
                            pretrained_window_sizes=[12, 12, 12, 6])
    return nn.ModuleList([net])

def build_all_net(config, init_trainloader):
    cls_net = build_swinv2(config, init_trainloader.num_classes)[0]
    spl_net = decoder(init_trainloader.dim_conf)
    net = nn.ModuleList([cls_net, spl_net])
    return net

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.MultiheadAttention):
        nn.init.xavier_normal_(m.in_proj_weight)
        nn.init.constant_(m.in_proj_bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    