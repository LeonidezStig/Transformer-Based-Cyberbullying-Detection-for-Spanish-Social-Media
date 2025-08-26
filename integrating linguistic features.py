# integrating linguistic features (like POS tags) with RoBERTa embeddings in a PyTorch model for cyberbullying detection. 

import torch 

import torch.nn as nn 

from transformers import AutoTokenizer, AutoModel 

import spacy 

 

class RoBERTaWithLinguisticFeatures(nn.Module): 

    def __init__(self, roberta_model_name='roberta-base-bne', pos_feature_dim=17, num_labels=2): 

        super().__init__() 

        self.roberta = AutoModel.from_pretrained(roberta_model_name) 

        # POS feature embedding layer (optional: treat pos tag indices as embeddings) 

        self.pos_embedding = nn.Embedding(pos_feature_dim, 16)  # example embedding size 16 

         

        # Classifier input dimension is RoBERTa hidden size + POS embedding size 

        roberta_hidden_size = self.roberta.config.hidden_size 

        self.classifier = nn.Sequential( 

            nn.Linear(roberta_hidden_size + 16, 128), 

            nn.ReLU(), 

            nn.Dropout(0.1), 

            nn.Linear(128, num_labels) 

        ) 

        # Load spaCy model for linguistic feature extraction 

        self.spacy_nlp = spacy.load("es_core_news_sm")  # or appropriate language model 

 

        # POS tag to index mapping - define your mapping based on spacy pos tags 

        self.pos2idx = {pos: idx for idx, pos in enumerate([ 

            'ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 

            'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X' 

        ])} 

 

    def extract_pos_tags(self, text): 

        doc = self.spacy_nlp(text) 

        pos_indices = [self.pos2idx.get(token.pos_, self.pos2idx['X']) for token in doc] 

        return pos_indices 

 

    def forward(self, input_texts): 

        # inputs: list of batch texts (strings) 

        batch_size = len(input_texts) 

         

        # Tokenize texts 

        encoded = self.roberta.tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt') 

        input_ids = encoded['input_ids'] 

        attention_mask = encoded['attention_mask'] 

 

        # Get RoBERTa embeddings 

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask) 

        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size) 

 

        # Extract POS tags and convert to tensor with padding to seq_len 

        pos_tensors = [] 

        max_len = sequence_output.size(1) 

        for text in input_texts: 

            pos_indices = self.extract_pos_tags(text) 

            # Pad or truncate to max_len 

            if len(pos_indices) < max_len: 

                pos_indices += [self.pos2idx['X']] * (max_len - len(pos_indices)) 

            else: 

                pos_indices = pos_indices[:max_len] 

            pos_tensors.append(torch.tensor(pos_indices)) 

        pos_tensor = torch.stack(pos_tensors).to(sequence_output.device)  # (batch_size, seq_len) 

 

        # Get POS embeddings 

        pos_embeds = self.pos_embedding(pos_tensor)  # (batch_size, seq_len, pos_embed_dim) 

 

        # Concatenate RoBERTa embeddings and POS embeddings 

        combined = torch.cat((sequence_output, pos_embeds), dim=2)  # (batch_size, seq_len, hidden+pos_dim) 

 

        # Aggregate token representations for classification, e.g., take CLS token (index 0) 

        combined_cls = combined[:, 0, :]  # (batch_size, hidden+pos_dim) 

 

        # Pass through classifier 

        logits = self.classifier(combined_cls)  # (batch_size, num_labels) 

 

        return logits 

 