import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        # Scale by head dimension for stable attention logits.
        wei = q @ k.transpose(-2,-1) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class NanoGPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

# Training Logic Wrapper
class LLMEngine:
    def __init__(self, text_data, config=None):
        self.text = text_data
        self.config = config or {
            'batch_size': 32,
            'block_size': 64,
            'max_iters': 100,
            'eval_interval': 20,
            'learning_rate': 1e-3,
            'n_embd': 128,
            'n_head': 4,
            'n_layer': 4,
            'dropout': 0.1,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        self.fallback_char = " " if " " in self.stoi else self.chars[0]
        
        self.model = NanoGPTModel(
            self.vocab_size, 
            self.config['n_embd'], 
            self.config['n_head'], 
            self.config['n_layer'], 
            self.config['block_size'], 
            self.config['dropout'],
            self.config['device']
        ).to(self.config['device'])
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        
        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(0.9*len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s):
        # Map unseen characters to a known fallback token to avoid hard failures.
        return [self.stoi.get(c, self.stoi[self.fallback_char]) for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        if len(data) < 2:
            raise ValueError("Training data is too short. Provide at least 2 characters of text.")

        current_block_size = min(self.config['block_size'], len(data) - 1)
        max_start = len(data) - current_block_size
        ix = torch.randint(max_start, (self.config['batch_size'],))
        x = torch.stack([data[i:i+current_block_size] for i in ix])
        y = torch.stack([data[i+1:i+current_block_size+1] for i in ix])
        return x.to(self.config['device']), y.to(self.config['device'])

    @torch.no_grad()
    def estimate_loss(self, eval_iters=20):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def train_step(self):
        xb, yb = self.get_batch('train')
        logits, loss = self.model(xb, yb)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def generate(self, prompt="", max_new_tokens=100):
        if not prompt:
            idx = torch.zeros((1, 1), dtype=torch.long, device=self.config['device'])
        else:
            idx = torch.tensor([self.encode(prompt)], dtype=torch.long, device=self.config['device'])
        
        self.model.eval()
        generated = self.model.generate(idx, max_new_tokens)
        self.model.train()
        return self.decode(generated[0].tolist())
