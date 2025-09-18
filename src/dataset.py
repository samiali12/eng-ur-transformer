import os 
import pandas as pd
import torch 

from pathlib import Path

from torch.utils.data import random_split, Dataset, DataLoader

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

def causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0

class BuildingDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tar, lang_src, lang_tar, seq_len):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tar = tokenizer_tar
        self.lang_src = lang_src
        self.lang_tar = lang_tar
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int32)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int32)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id(['PAD'])], dtype=torch.int32)

    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):
        pair = self.dataset.iloc[index]
        src_text = pair[self.lang_src]
        tar_text = pair[self.lang_tar]

        input_token = self.tokenizer_src.encoder(src_text).id
        output_token = self.tokenizer_tar.encode(tar_text).id 
        
        input_token_pad = self.seq_len - len(input_token) - 2
        output_token_pad = self.seq_len - len(output_token) - 1

        if input_token_pad < 0 or output_token_pad < 0:
            return ValueError("Sentence is too long")
        
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(input_token, dtype=torch.int32),
            self.eos_token,
            torch.tensor([self.pad_token] * input_token_pad, dtype=torch.int32)
        ])

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(output_token, dtype=torch.int32),
            torch.tensor([self.pad_token] * output_token_pad, dtype=torch.int32)
        ])

        label = torch.cat([
        torch.tensor(output_token, dtype=torch.int64),
        self.eos_token,
        torch.tensor([self.pad_token] * output_token_pad, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'label': label,
        }




def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "../" "data")

    english_file = os.path.join(data_dir, "english.txt")
    urdu_file = os.path.join(data_dir, "urdu.txt")

    with open(english_file, "r", encoding="utf-8") as f:
        english_corpus = [line.strip() for line in f]
        english_corpus = list(english_corpus)

    with open(urdu_file, "r", encoding="utf-8") as f:
        urdu_corpus = [line.strip() for line in f]
        urdu_corpus = list(urdu_corpus)

    data = pd.DataFrame({'english': english_corpus, 'urdu': urdu_corpus})
    return data



def build_tokenizer(config, data, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        language = data[lang]
        tokenizer = Tokenizer(WordLevel(unk_token='UNK'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'])
        tokenizer.train_from_iterator(language, trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)

    return tokenizer



def build_data(config):
    data = load_data()
    tokenizer_source = build_tokenizer(config, data, config['lang_src'])
    tokenizer_target = build_data(config, data, config['lang_tar'])

    total_size = len(data)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.2)

    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    train_data = BuildingDataset(train_dataset, tokenizer_source, tokenizer_target, config['lang_src'], config['lang_tar'], config['seq_len'])
    val_data = BuildingDataset(val_dataset, tokenizer_source, tokenizer_target, config['lang_src'], config['lang_tar'], config['seq_len'])

    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader



data = load_data()
tokenizer = build_tokenizer({'tokenizer_file': 'english.json'}, data, 'english')