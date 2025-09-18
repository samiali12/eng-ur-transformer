import torch
from train import train_model

def get_config():
  config = {
      'batch_size': 8,
      'num_epochs': 10,
      'lr': 10**-4,
      'seq_len': 350,
      'd_model': 512,
      'lang_src': 'english',
      'lang_tar': 'urdu',
      'model_folder': 'weights',
      'model_basename': 'tmodel_',
      'preload': None,
      'tokenizer_file': 'tokenizer_{0}.json',
      'experiment_name': 'runs/tmodel',
      'ffn_hiddn_dim': 1048,
      'num_heads': 8,
      'dropout': 0.1,
      'device': 'cuda' if torch.cuda.is_available() else 'cpu'
  }
  return config


if __name__ == '__main__':
    config = get_config()
    train_model(config)