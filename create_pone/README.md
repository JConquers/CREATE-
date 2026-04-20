# CREATE-Pone (CREATE++ Signed Variant)

This folder contains a standalone CREATE-Pone implementation that combines:

- CREATE-style cross-representation training (sequential + graph alignment)
- Pone-GNN-style signed graph modeling (separate positive and negative branches)
- CREATE++ losses and two-phase optimization protocol

## Structure

- `create_pone/dataset/`
  - `loader.py`: dataset loading from existing `dataset_loaders`
  - `sequence_dataset.py`: user sequence dataset + collator
  - `signed_graph.py`: signed graph builder + triplet sampler
- `create_pone/models/`
  - `signed_gnn.py`: dual-branch signed GNN
  - `sequence_encoder.py`: SASRec-style causal transformer
  - `create_pone.py`: integrated model
- `create_pone/losses.py`: CREATE++ local/global/alignment losses
- `create_pone/trainer.py`: training pipeline
- `run_create_pone.py`: CLI entrypoint

## Run

Beauty:

python run_create_pone.py --dataset beauty --data-dir ./data --epochs 30 --warmup-epochs 5

Books:

python run_create_pone.py --dataset books --data-dir ./data --epochs 30 --warmup-epochs 5
