from yacs.config import CfgNode as CN

_C = CN()



def update_specific_config(cfg, name):
    cfg.defrost()
    cfg.merge_from_file(name)
    cfg.freeze()

def update_config(cfg, args):
    cfg.defrost() # Makes the cfg mutable
    cfg.merge_from_file(args.cfg) # Loads input cfg file to cfg
    cfg.freeze() # Turns cfg immutable
