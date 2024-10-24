import pandas as pd 
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast
from torch.nn.utils.rnn import pad_sequence
import torch
from tqdm import tqdm

class get_data:
    def __init__(self, bs=32):
        df = pd.read_csv('fra.txt', delimiter='\t', names=['english', 'french', 'attribute'], usecols=['english', 'french'])
        df.reset_index(drop=True)
        df.head()

        en = df['english']
        fr = df['french']

        english_tokenizer = SentencePieceBPETokenizer()
        english_tokenizer.train_from_iterator(
            df['english'],
            vocab_size=30_000,
            min_frequency=2,
            show_progress=True,
            limit_alphabet=500,
        )
        english_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=english_tokenizer
        )
        self.eng_eng_encoded = []
        for i in tqdm(range(len(df['english']))):
            self.eng_eng_encoded.append(torch.tensor(english_tokenizer.encode(df['english'].iloc[i]), dtype=torch.long))

        french_tokenizer = SentencePieceBPETokenizer()
        french_tokenizer.train_from_iterator(
            df['french'],
            vocab_size=30_000,
            min_frequency=5,
            show_progress=True,
            limit_alphabet=500,
        )
        french_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=french_tokenizer
        )
        self.french_eng_encoded = []
        for i in tqdm(range(len(df['french']))):
            self.french_eng_encoded.append(torch.tensor(french_tokenizer.encode(df['french'].iloc[i]), dtype=torch.long))

        self.eng_data, self.fre_data = self._get()
        self.index = 0
        self.bs = bs

    def _get(self):
        t = self.eng_eng_encoded + self.french_eng_encoded
        a = pad_sequence(t, batch_first=True)
        eng, fre = a[:len(self.eng_eng_encoded)], a[len(self.eng_eng_encoded):]
        return eng, fre

    def __next__(self):
        if self.index * self.bs < len(self.eng_data):
            self.index+=1
            return self.eng_data[self.index*self.bs:(self.index+1)*self.bs], self.fre_data[self.index*self.bs:(self.index+1)*self.bs]
        else:
            raise StopIteration