import os
import torch
import torch.nn as nn

from dataset import build_data
from model import build_transformer

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

def get_weights_file_path(config, epoch: str):
  model_folder = config['model_folder']
  model_basename = config['model_basename']
  model_filename = f"{model_basename}{epoch}.pt"
  return os.path.join(model_folder, model_filename)

def get_model(config, src_vocab_size, tar_vocab_size):
    transformer = build_transformer(
        src_vocab_size,
        tar_vocab_size,
        config['seq_len'],
        config['seq_len'],
        config['d_model'],
        config['ffn_hiddn_dim'],
        config['num_heads'],
        config['dropout']
    )
    return transformer

def train_model(config):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device {device}')

    train_dataLoader, val_dataloader, src_tokenizer, tar_tokenizer  = build_data(config)

    model = get_model(config, src_tokenizer.get_vocab_size(), tar_tokenizer.get_vocab_size()).to(device)
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state_dict=state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataLoader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, tar_tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})

            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

