from yacs.config import CfgNode as CN

_C = CN()

_C.ROOT_DIR = ''
_C.EXP_NAME = 'default'
_C.SEED = 0
_C.CHECKPOINT_FREQ = 5000 #TODO PICK A REASONABLE VALUE
_C.LOGGING_FREQ = 5000 #TODO PICK A REASONABLE VALUE
_C.DEBUG = False

_C.ENVIRONMENT = CN()
_C.ENVIRONMENT.NAME = ''
_C.ENVIRONMENT.NUM_ENV = ''
_C.ENVIRONMENT.TOTAL_TIMESTEPS = ''

_C.RL = CN()
_C.RL.ALGO = 'ppo'
_C.RL.POLICY = 'mlppolicy'
_C.RL.BATCH_SIZE = 2048
_C.RL.LR = 3e-4
_C.RL.ARGS = '{}'



def update_specific_config(cfg, name):
    cfg.defrost()
    cfg.merge_from_file(name)
    cfg.freeze()

def update_config(cfg, args):
    cfg.defrost() # Makes the cfg mutable
    cfg.merge_from_file(args.cfg) # Loads input cfg file to cfg
    cfg.freeze() # Turns cfg immutable
