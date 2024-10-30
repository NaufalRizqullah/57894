import yaml

def load_config(path):
    """
    Load configuration from a yaml file.

    This function reads the configuration from a yaml file.

    Returns
    -------
    config : dict
        A dictionary containing the configuration loaded from the yaml file.
    """
    with open(path) as p:
        config = yaml.safe_load(p)
    
    return config

def test_load_config():
    config = load_config()

    assert config is not None
    assert 'epoch' in config
    assert config['epoch'] == 200
    assert 'batch_size' in config
    assert config['batch_size'] == 16
    assert 'image_channels' in config
    assert config['image_channels'] == 3
    assert 'learning_rate' in config
    assert config['learning_rate'] == '1e-5'
    assert 'lambda_cycle' in config
    assert config['lambda_cycle'] == 10
    assert 'lambda_identity' in config
    assert config['lambda_identity'] == 0.0
    assert 'summer_path' in config
    assert config['summer_path'] == '/kaggle/input/summer2winter-yosemite/trainA'
    assert 'winter_path' in config
    assert config['winter_path'] == '/kaggle/input/summer2winter-yosemite/trainB'
    assert 'display_step' in config
    assert config['display_step'] == 500
    assert 'load_checkpoint' in config
    assert not config['load_checkpoint']
    assert 'ckpt_path' in config
    assert config['ckpt_path'] == ''

if __name__ == '__main__':
    test_load_config()

