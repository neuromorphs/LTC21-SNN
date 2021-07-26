import yaml

def load_config(filepath, verbose=True):
    """Loads a yaml file and returns its contents"""
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    if verbose:
        print("loaded config file:")
        for k, v in data.items():
            print(f"{k:22s} : {str(v):20s}  {type(v)}")
        print()
    return data

