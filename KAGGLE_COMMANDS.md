# Kaggle Training Commands Template

Quick reference for running CREATE-PONE on Kaggle GPUs.

---

## Quick Start

### Step 1: Create Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Enable GPU: Settings → Accelerator → **GPU T4 x2** or **GPU P100**
4. Turn on Internet: Settings → Internet → **On** (for downloading datasets)

### Step 2: Add Data (Optional but Recommended)

To skip dataset download time (5-10 minutes), upload data beforehand:

1. Click "Add Data" → "New Dataset"
2. Upload your Amazon Books/Beauty CSV files
3. Note the dataset path (e.g., `/kaggle/input/amazon-books`)

---

## Dataset Caching (Automatic Optimization)

### How Caching Works

The code **automatically caches** preprocessed data so you never need to reprocess:

```
First run:
  Download Books.csv.gz → Preprocess → Save books_processed.pkl (5-10 min)

Second run:
  books_processed.pkl exists → Load cached data (~2 seconds) ✅
```

### Cache File Locations

| Dataset | Raw File | Cached File |
|---------|----------|-------------|
| Books | `./data/Books.csv.gz` | `./data/books_processed.pkl` |
| Beauty | `./data/Beauty_and_Personal_Care.csv.gz` | `./data/beauty_processed.pkl` |

### What's Cached

The `.pkl` file stores:
- `train_df`, `val_df`, `test_df` - Split DataFrames
- `num_users`, `num_items` - Counts
- `user2items`, `item2users` - Mappings
- `user_sequences` - Sequential user histories

### Kaggle-Specific: Persist Cache Across Sessions

**Important:** Kaggle resets `./data/` when sessions restart. Use these strategies:

#### Strategy 1: Copy Cache to Persistent Storage (Recommended)

After first run, save the cache:

```python
# Run after first training completes
!mkdir -p /kaggle/working/processed_data
!cp ./data/books_processed.pkl /kaggle/working/processed_data/
!cp ./data/Books.csv.gz /kaggle/working/processed_data/  # Optional: skip re-download
```

In subsequent runs, restore before training:

```python
# Run BEFORE training
!mkdir -p ./data
!cp /kaggle/working/processed_data/books_processed.pkl ./data/
!cp /kaggle/working/processed_data/Books.csv.gz ./data/  # Optional
```

#### Strategy 2: Upload Preprocessed Data as Kaggle Dataset (Fastest)

1. Run locally once to generate `books_processed.pkl`
2. Upload `books_processed.pkl` as a Kaggle Dataset
3. In Kaggle notebook:

```python
!mkdir -p ./data
!cp /kaggle/input/your-preprocessed-data/books_processed.pkl ./data/
```

**Time saved:** ~10 minutes per session (no download, no preprocessing)

#### Strategy 3: Full Notebook with Cache Restore

```python
# ============================================================================
# CELL 1: Setup and Dependencies
# ============================================================================
!pip install torch-geometric

# ============================================================================
# CELL 2: Restore Cached Preprocessed Data (Skip if first run)
# ============================================================================
import os
os.makedirs('./data', exist_ok=True)

# Check if cache exists in persistent storage
if os.path.exists('/kaggle/working/processed_data/books_processed.pkl'):
    print("Restoring cached preprocessed data...")
    !cp /kaggle/working/processed_data/books_processed.pkl ./data/
    !cp /kaggle/working/processed_data/Books.csv.gz ./data/
    print("Cache restored! Skipping download and preprocessing.")
else:
    print("No cache found. Will download and preprocess (first run only).")

# ============================================================================
# CELL 3: Verify GPU is Available
# ============================================================================
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# CELL 4: Run Training
# ============================================================================
%run train_kaggle.py \
    --dataset books \
    --pretrain_epochs 150 \
    --num_epochs 300 \
    --embedding_dim 64 \
    --batch_size 256 \
    --gpu 0 \
    --seed 42

# ============================================================================
# CELL 5: Save Cache for Future Sessions (After Training Completes)
# ============================================================================
!mkdir -p /kaggle/working/processed_data
!cp ./data/books_processed.pkl /kaggle/working/processed_data/
!cp ./data/Books.csv.gz /kaggle/working/processed_data/
print("Cache saved for future sessions!")
```

### Step 3: Copy Code to Kaggle

**Option A: Upload files directly**
- Upload all `.py` files and folders to Kaggle notebook

**Option B: Clone from GitHub**
```python
!git clone https://github.com/your-username/CREATE-.git
%cd CREATE-
```

### Step 4: Install Dependencies

```python
# Required for PyTorch Geometric on Kaggle
!pip install torch-geometric
```

---

## Training Commands

### Full CREATE-PONE Training (Two-Stage)

```bash
%run train_kaggle.py \
    --dataset books \
    --data_dir ./data \
    --pretrain_epochs 150 \
    --num_epochs 300 \
    --embedding_dim 64 \
    --sasrec_heads 4 \
    --sasrec_layers 2 \
    --ponegnn_layers 2 \
    --fusion_type concat \
    --batch_size 256 \
    --lr 1e-3 \
    --barlow_weight 0.01 \
    --orthogonal_weight 0.01 \
    --contrastive_weight 0.1 \
    --top_k 10 \
    --eval_every 10 \
    --gpu 0 \
    --seed 42
```

### Quick Test Run (Debug Mode)

```bash
%run train_kaggle.py \
    --dataset books \
    --data_dir ./data \
    --pretrain_epochs 10 \
    --num_epochs 20 \
    --embedding_dim 32 \
    --batch_size 128 \
    --gpu 0 \
    --no_save
```

### Beauty Dataset

```bash
%run train_kaggle.py \
    --dataset beauty \
    --data_dir ./data \
    --pretrain_epochs 150 \
    --num_epochs 300 \
    --embedding_dim 64 \
    --batch_size 256 \
    --lr 1e-3 \
    --gpu 0
```

### PoneGNN Pre-training Only (Stage 1)

```bash
%run train_kaggle.py \
    --dataset books \
    --data_dir ./data \
    --pretrain_epochs 150 \
    --embedding_dim 64 \
    --ponegnn_layers 2 \
    --batch_size 256 \
    --lr 1e-3 \
    --gpu 0
```

---

## Complete Kaggle Notebook Template

Copy this entire template into a Kaggle notebook:

```python
# ============================================================================
# CELL 1: Setup and Dependencies
# ============================================================================
!pip install torch-geometric

# ============================================================================
# CELL 2: Copy Data (if uploaded as Kaggle dataset)
# ============================================================================
import os
os.makedirs('./data', exist_ok=True)

# If you uploaded Amazon Books dataset to Kaggle:
# !cp -r /kaggle/input/amazon-books/* ./data/

# Or download automatically (takes 5-10 minutes):
# The script handles this automatically

# ============================================================================
# CELL 3: Verify GPU is Available
# ============================================================================
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# CELL 4: Run Training
# ============================================================================
# Full CREATE-PONE training
%run train_kaggle.py \
    --dataset books \
    --pretrain_epochs 150 \
    --num_epochs 300 \
    --embedding_dim 64 \
    --sasrec_heads 4 \
    --sasrec_layers 2 \
    --ponegnn_layers 2 \
    --fusion_type concat \
    --batch_size 256 \
    --lr 1e-3 \
    --barlow_weight 0.01 \
    --orthogonal_weight 0.01 \
    --contrastive_weight 0.1 \
    --top_k 10 \
    --eval_every 10 \
    --gpu 0 \
    --seed 42
```

---

## GPU Usage on Kaggle

### Yes, GPU is Supported! ✅

The code **already supports GPU training** with no changes needed. Just:

1. **Enable GPU in Kaggle Settings:**
   - Click "Settings" (right panel)
   - Under "Accelerator", select: **GPU T4 x2** or **GPU P100**
   - T4 has 16GB VRAM, P100 has 16GB VRAM

2. **Use `--gpu 0` flag:**
   - The script automatically detects and uses GPU
   - `--gpu 0` means "use first GPU"
   - If running on CPU, use `--gpu -1`

3. **Verify GPU is active:**
```python
import torch
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.get_device_name(0))  # e.g., "Tesla T4"
```

### GPU Memory Considerations

| Batch Size | Approx. VRAM Usage | Recommended For |
|------------|-------------------|-----------------|
| 128 | ~2-4 GB | Debug/testing |
| 256 | ~4-8 GB | Standard training |
| 512 | ~8-12 GB | Large datasets |
| 1024 | ~12-16 GB | Maximum throughput |

If you get OOM (Out of Memory), reduce `--batch_size`.

---

## Mini-Batch Training Paradigm

### Yes, Mini-Batch Training is Used ✅

The code follows mini-batch training in **both stages**:

#### Stage 1: PoneGNN Pre-training

```python
# Mini-batch iteration over users
users = list(sampler.user_items.keys())
random.shuffle(users)

for batch_start in range(0, len(users), batch_size):
    batch_users = users[batch_start:batch_start + batch_size]
    
    for user_id in batch_users:
        # Process user's positive items
        for item_id in pos_items:
            # Single sample gradient update
            loss = model.compute_loss(...)
            loss.backward()
            optimizer.step()
```

**Note:** Stage 1 processes users in batches but updates weights per sample (stochastic). This is intentional for graph embedding learning.

#### Stage 2: Joint Training (True Mini-Batch)

```python
# DataLoader provides proper mini-batches
for batch in train_loader:
    item_sequences = batch['padded_sequence_ids'].to(device)  # (batch_size, seq_len)
    mask = batch['mask'].to(device)                            # (batch_size, seq_len)
    labels = batch['labels.ids'].to(device)                    # (batch_size,)
    users = batch['user.ids'].to(device)                       # (batch_size,)
    
    # Single forward/backward pass for entire batch
    loss_dict = model.compute_joint_loss(...)
    loss = loss_dict['total_loss']
    loss.backward()
    optimizer.step()
```

**Stage 2 uses true mini-batch training** with batch size = 256 (default).

---

## Parameter Reference

| Parameter | Default | Description | When to Change |
|-----------|---------|-------------|----------------|
| `--dataset` | books | Dataset name | Use `beauty` for Beauty dataset |
| `--data_dir` | ./data | Data directory path | Change if data is elsewhere |
| `--pretrain_epochs` | 50 | Stage 1 epochs | Increase to 150 for full training |
| `--num_epochs` | 100 | Stage 2 epochs | Increase to 300 for full training |
| `--embedding_dim` | 64 | Embedding size | 32 for testing, 64-128 for production |
| `--batch_size` | 256 | Training batch size | Reduce if OOM, increase for speed |
| `--lr` | 1e-3 | Learning rate | Keep as-is, or try 5e-4 |
| `--gpu` | 0 | GPU device ID | -1 for CPU, 0 for first GPU |
| `--seed` | 42 | Random seed | Change for different initializations |
| `--eval_every` | 10 | Evaluation frequency | Reduce to 5 for more frequent logging |
| `--top_k` | 10 | Top-K metrics | Keep as-is |

---

## Expected Training Output

```
Using GPU: Tesla T4

Loading dataset: books
Dataset statistics:
  num_users: 50,123
  num_items: 87,456
  num_interactions: 1,234,567
  sparsity: 0.9997

============================================================
Stage 1: Pre-training PoneGNN Graph Encoder
============================================================
Epoch   1/150 | Loss: 0.8234
Epoch  10/150 | Loss: 0.5123
...
Pre-training complete. Best loss: 0.4521

Creating CREATE++ model with pre-trained graph encoder...
Loaded pre-trained PoneGNN weights.

============================================================
Stage 2: Joint Training of SASRec + PoneGNN
============================================================
Epoch   1/300 | Loss: 1.2345 | HR@10: 0.1234 | NDCG@10: 0.0876
Epoch  10/300 | Loss: 0.8765 | HR@10: 0.2345 | NDCG@10: 0.1543
...
Joint training complete.
Best HR@10: 0.3456
Best NDCG@10: 0.2345

Final model saved to: ./outputs/create++_books_20240101_120000/create_plus_plus_final.pt
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size
```bash
--batch_size 128  # or even 64
```

### Issue: "No module named 'torch_geometric'"

**Solution:** Install with specific version for Kaggle
```python
!pip install torch-geometric==2.4.0
```

### Issue: Dataset download timeout

**Solution 1:** Use cached preprocessed data (see "Dataset Caching" section above)

**Solution 2:** Upload data files to Kaggle beforehand and copy:
```python
!cp -r /kaggle/input/your-dataset/* ./data/
```

**Solution 3:** Enable Internet in Kaggle Settings and let the script auto-download (first run only, then cached)

### Issue: Training is slow

**Solutions:**
1. Ensure GPU is enabled (check Settings → Accelerator)
2. Increase batch size if VRAM allows
3. Reduce `--eval_every` to log less frequently
4. Use fewer epochs for debugging

---

## Kaggle Session Limits

| Resource | Limit | Notes |
|----------|-------|-------|
| GPU Session | 12 hours | Notebook will reset after |
| CPU Session | 24 hours | |
| RAM | 16 GB | Monitor usage |
| Disk | 80 GB | Enough for datasets + checkpoints |
| Weekly GPU Hours | ~30 hours | Varies by account tier |

**Tip:** Save checkpoints frequently! Kaggle sessions expire.

```bash
# Save checkpoints every 50 epochs
# (handled automatically by --save_checkpoint flag)
```

---

## Saving and Loading Checkpoints

### Checkpoints are Auto-Saved

```
./outputs/create++_books_YYYYMMDD_HHMMSS/
├── ponegnn_pretrain.pt      # After Stage 1
└── create_plus_plus_final.pt # After Stage 2
```

### Load Checkpoint for Inference

```python
import torch
from models import CREATEPlusPlusModel

# Load checkpoint
checkpoint = torch.load('./outputs/.../create_plus_plus_final.pt')

# Create model with same config
model = CREATEPlusPlusModel(
    num_users=50123,
    num_items=87456,
    embedding_dim=64,
    sasrec_heads=4,
    sasrec_layers=2,
    ponegnn_layers=2,
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```
