import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 512  
EPOCHS = 5  # Changed as recommended to speed up training
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    
    def __init__(self, skipgram_df):
        
        self.center_words = torch.LongTensor(skipgram_df['center'].values)
        self.context_words = torch.LongTensor(skipgram_df['context'].values)
    
    def __len__(self):
        return len(self.center_words)
    
    def __getitem__(self, idx):
        return self.center_words[idx], self.context_words[idx]


class Word2Vec(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.center_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.context_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
    
    def forward(self, center_words, context_words):
        
        center_embeds = self.center_embeddings(center_words)  
        context_embeds = self.context_embeddings(context_words)
        
        scores = torch.sum(center_embeds * context_embeds, dim=1)
        
        return scores
    
    def get_embeddings(self):
        
        return self.center_embeddings.weight.data.cpu().numpy()


# Load processed data
print("Loading processed data...")
with open('/Users/nolanwhite/Documents/GitHub/stat359/instructor/Assignment_2/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

vocab_size = len(data['word2idx'])
print(f"Vocabulary size: {vocab_size}")
print(f"Number of skip-gram pairs: {len(data['skipgram_df'])}")

counter = data['counter']

word_counts = torch.zeros(vocab_size)
for idx in range(vocab_size):
    word = data['idx2word'][idx]
    word_counts[idx] = counter.get(word, 0)

print(f"Total word count: {word_counts.sum().item()}")

word_freqs = word_counts ** 0.75
word_freqs = word_freqs / word_freqs.sum()

print(f"Negative sampling distribution created")

# Device selection: CUDA > MPS > CPU

# I have a mac so I think I needed MPS, but wasn't sure how to move to GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS (Metal Performance Shaders)")
else:
    device = torch.device('cpu')
    print("Using CPU")

word_freqs_cpu = word_freqs.cpu()

def sample_negative_words_batch(positive_context, num_samples, word_freqs):
    
    batch_size = positive_context.size(0)
    
    oversample_factor = 2
    total_samples = batch_size * num_samples * oversample_factor
    
    all_samples = torch.multinomial(word_freqs, total_samples, replacement=True)
    
    all_samples = all_samples.view(batch_size, -1)
    
    negative_samples = []
    for i in range(batch_size):
        valid_samples = all_samples[i][all_samples[i] != positive_context[i]]
        if len(valid_samples) >= num_samples:
            negative_samples.append(valid_samples[:num_samples])
        else:
    
            additional = torch.multinomial(word_freqs, num_samples, replacement=True)
            negative_samples.append(additional)
    
    return torch.stack(negative_samples)


dataset = SkipGramDataset(data['skipgram_df'])
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Loss Function
criterion = nn.BCEWithLogitsLoss(reduction='none')

def make_targets(center, context, vocab_size):
    """Helper function to create targets."""
    batch_size = center.size(0)
    targets = torch.zeros(batch_size, vocab_size)
    targets.scatter_(1, context.unsqueeze(1), 1)
    return targets

# Training loop
print("\nStarting training...")
print(f"Batch size: {BATCH_SIZE}, Batches per epoch: {len(dataloader)}, Total epochs: {EPOCHS}")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    
    for center_words, context_words in progress_bar:
        batch_size = center_words.size(0)
        
        
        negative_words = sample_negative_words_batch(context_words, NEGATIVE_SAMPLES, word_freqs_cpu)
        
        
        center_words = center_words.to(device)
        context_words = context_words.to(device)
        negative_words = negative_words.to(device)
        
        
        optimizer.zero_grad()
        
        
        positive_scores = model(center_words, context_words)
        positive_labels = torch.ones_like(positive_scores)
        
    
        center_expanded = center_words.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES).contiguous().view(-1)
        negative_flat = negative_words.view(-1)
        
        negative_scores = model(center_expanded, negative_flat)  
        negative_labels = torch.zeros_like(negative_scores)
        
        positive_loss = criterion(positive_scores, positive_labels)  
        negative_loss = criterion(negative_scores, negative_labels)
        
        negative_loss = negative_loss.view(batch_size, NEGATIVE_SAMPLES).sum(dim=1)
        

        per_example_loss = positive_loss + negative_loss 
        
        loss = per_example_loss.mean()
        
        loss.backward()
        optimizer.step()
        

        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        

        del center_words, context_words, negative_words
        del positive_scores, negative_scores
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / num_batches
    print(f'Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}')

# Save embeddings and mappings
print("\nSaving embeddings...")
embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({
        'embeddings': embeddings, 
        'word2idx': data['word2idx'], 
        'idx2word': data['idx2word']
    }, f)
print("Embeddings saved to word2vec_embeddings.pkl")
print(f"Embedding shape: {embeddings.shape}")