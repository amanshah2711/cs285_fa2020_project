from yacs.config import CfgNode as CN

_C = CN()

_C.ROOT_DIR = ''
_C.EXP_NAME = 'default'
_C.SEED = 0
_C.CHECKPOINT_INTERVAL = 131072
_C.LOG_INTERVAL = 2048
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
_C.RL.NORMALIZE = True
_C.RL.NORMALIZE_OBSERVATIONS = True
_C.RL.ARGS = '{}'

_C.EMBED.INDEX  = 0
_C.EMBED.PATH = 1
_C.EMBED.TYPE = "zoo"
_C.EMBED.NOISE = False
_C.EMBED.NOISE = '{}'
_C.EMBED.MASK = False
_C.EMBED.MASK_KWARGS = '{"masking_type": "initialization"}'


def update_config(cfg, args):
    cfg.defrost() # Makes the cfg mutable
    cfg.merge_from_file(args.cfg) # Loads input cfg file to cfg
    cfg.freeze() # Turns cfg immutable
