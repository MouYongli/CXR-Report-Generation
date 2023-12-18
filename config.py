import yaml

def get_config(args, cfg_path):
    with open(cfg_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    for key, value in yaml_cfg.items():
                if isinstance(value, list):
                    for v in value:
                        getattr(args, key.lower(), []).append(v)
                else:
                    setattr(args, key.lower(), value)
    return args