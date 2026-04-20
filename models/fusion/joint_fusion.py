"""Joint fusion module for CREATE++ Pone variant.

This module implements the joint training strategy from the CREATE++ paper
with four loss components:
1. Sequential Loss (SASRec) - Cross-entropy for next-item prediction
2. Dual-Feedback Loss - Separate BPR for positive/negative rated items
3. Barlow Twins Alignment Loss - Redundancy reduction between encoders
4. Orthogonal Loss - Decorrelate positive/negative graph embeddings
Plus: Contrastive Loss from Pone-GNN for cross-space alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointFusionModule(nn.Module):
    """Fusion module for combining sequential and graph embeddings.

    Supports multiple fusion strategies:
    - concat: Concatenate and project
    - sum: Element-wise sum
    - gate: Learned gating mechanism
    - mlp: MLP fusion
    """

    def __init__(
        self,
        embedding_dim: int,
        fusion_type: str = 'concat',
        hidden_dim: int = None,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.embedding_dim = embedding_dim

        if fusion_type == 'concat':
            self.fusion_dim = embedding_dim * 2
            self.output_proj = nn.Linear(self.fusion_dim, embedding_dim)
        elif fusion_type == 'sum':
            self.fusion_dim = embedding_dim
            self.output_proj = nn.Identity()
        elif fusion_type == 'gate':
            self.fusion_dim = embedding_dim
            self.gate = nn.Linear(embedding_dim * 2, embedding_dim)
            self.output_proj = nn.Identity()
        elif fusion_type == 'mlp':
            hidden_dim = hidden_dim or embedding_dim * 2
            self.fusion_dim = embedding_dim * 2
            self.mlp = nn.Sequential(
                nn.Linear(self.fusion_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, embedding_dim),
            )
            self.output_proj = nn.Identity()
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, sequential_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        """Fuse sequential and graph embeddings.

        Args:
            sequential_emb: Embeddings from SASRec encoder (batch, dim)
            graph_emb: Embeddings from PoneGNN encoder (batch, dim)

        Returns:
            Fused embeddings (batch, dim)
        """
        if self.fusion_type == 'concat':
            combined = torch.cat([sequential_emb, graph_emb], dim=-1)
            fused = self.output_proj(combined)
        elif self.fusion_type == 'sum':
            fused = sequential_emb + graph_emb
        elif self.fusion_type == 'gate':
            combined = torch.cat([sequential_emb, graph_emb], dim=-1)
            gate = torch.sigmoid(self.gate(combined))
            fused = gate * sequential_emb + (1 - gate) * graph_emb
        elif self.fusion_type == 'mlp':
            combined = torch.cat([sequential_emb, graph_emb], dim=-1)
            fused = self.mlp(combined)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        return fused


class CREATEPlusPlusModel(nn.Module):
    """CREATE++ Pone variant model with joint SASRec + PoneGNN training.

    Implements all five loss components:
    1. Sequential Loss (SASRec cross-entropy) - learns temporal patterns
    2. Dual-Feedback Loss (positive/negative BPR) - learns from ratings
    3. Barlow Twins Alignment Loss (redundancy reduction) - aligns encoders
    4. Orthogonal Loss (embedding decorrelation) - decorrelates pos/neg
    5. Contrastive Loss (InfoNCE-style) - cross-space alignment

    Args:
        num_users: Number of users
        num_items: Number of items
        embedding_dim: Embedding dimension
        sasrec_heads: Number of attention heads in SASRec
        sasrec_layers: Number of transformer layers in SASRec
        ponegnn_layers: Number of graph convolution layers in PoneGNN
        max_sequence_length: Maximum sequence length
        fusion_type: Fusion strategy ('concat', 'sum', 'gate', 'mlp')
        fusion_hidden_dim: Hidden dimension for MLP fusion
        dim_feedforward: Feedforward dimension in SASRec transformer
        dropout: Dropout rate
        reg: L2 regularization
        barlow_weight: Weight for Barlow Twins loss
        orthogonal_weight: Weight for orthogonal loss
        dual_feedback_weight: Weight for dual-feedback loss
        contrastive_weight: Weight for contrastive loss
        temperature: Temperature for contrastive loss
        contrastive_interval: Apply contrastive loss every N epochs
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        sasrec_heads: int = 4,
        sasrec_layers: int = 2,
        ponegnn_layers: int = 2,
        max_sequence_length: int = 50,
        fusion_type: str = 'concat',
        fusion_hidden_dim: int = 128,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        reg: float = 1e-4,
        barlow_weight: float = 0.01,
        orthogonal_weight: float = 0.01,
        dual_feedback_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        temperature: float = 1.0,
        contrastive_interval: int = 10,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        from models.encoders.sequential_encoder import SASRecEncoder
        from models.encoders.graph_encoder import PoneGNNEncoder

        # Sequential encoder (SASRec)
        self.sequential_encoder = SASRecEncoder(
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_heads=sasrec_heads,
            num_layers=sasrec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
        )

        # Graph encoder (PoneGNN) - with contrastive learning support
        self.graph_encoder = PoneGNNEncoder(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=ponegnn_layers,
            reg=reg,
            temperature=temperature,
            contrastive_weight=contrastive_weight,
        )

        # Fusion module
        self.fusion_module = JointFusionModule(
            embedding_dim=embedding_dim,
            fusion_type=fusion_type,
            hidden_dim=fusion_hidden_dim,
        )

        # Contrastive loss interval
        self.contrastive_interval = contrastive_interval

        # Output projection
        self.output_projection = nn.Linear(embedding_dim, num_items + 1)

        # Loss weights
        self.barlow_weight = barlow_weight
        self.orthogonal_weight = orthogonal_weight
        self.dual_feedback_weight = dual_feedback_weight
        self.contrastive_weight = contrastive_weight

        self._init_weights()

    def _init_weights(self):
        """Initialize output projection weights."""
        nn.init.xavier_normal_(self.output_projection.weight)

    def forward(
        self,
        item_sequences: torch.Tensor,
        mask: torch.Tensor,
        pos_edge_index: torch.Tensor = None,
        neg_edge_index: torch.Tensor = None,
        training_graph: bool = False,
    ) -> dict:
        """
        Forward pass for CREATE++ model.

        Args:
            item_sequences: Item sequences (batch_size, seq_len)
            mask: Boolean mask for valid positions (batch_size, seq_len)
            pos_edge_index: Positive edge indices (2, num_pos_edges)
            neg_edge_index: Negative edge indices (2, num_neg_edges)
            training_graph: Whether to use graph encoder

        Returns:
            Dictionary containing:
            - sequential_scores: Scores from sequential encoder
            - sequential_emb: Sequential embeddings
            - user_pos_emb: User positive graph embeddings (if graph enabled)
            - user_neg_emb: User negative graph embeddings (if graph enabled)
            - item_pos_emb: Item positive graph embeddings (if graph enabled)
            - fused_emb: Fused embeddings
            - fused_scores: Scores from fused embeddings
        """
        # Get sequential embeddings
        seq_output = self.sequential_encoder(item_sequences, mask)
        sequential_emb = seq_output['user_embedding']

        result = {
            'sequential_scores': seq_output['item_scores'],
            'sequential_emb': sequential_emb,
        }

        if training_graph and pos_edge_index is not None:
            # Get graph embeddings (both positive and negative)
            pos_emb, neg_emb = self.graph_encoder(pos_edge_index, neg_edge_index)

            # Get graph embeddings for users in batch
            users = item_sequences[:, 0]
            user_pos_emb = pos_emb[:self.num_users][users]
            user_neg_emb = neg_emb[:self.num_users][users]
            item_pos_emb = pos_emb[self.num_users:]

            # Fuse embeddings (using positive graph embeddings)
            fused_emb = self.fusion_module(sequential_emb, user_pos_emb)

            # Compute scores
            fused_scores = self.output_projection(fused_emb)

            result['user_pos_emb'] = user_pos_emb
            result['user_neg_emb'] = user_neg_emb
            result['item_pos_emb'] = item_pos_emb
            result['fused_emb'] = fused_emb
            result['fused_scores'] = fused_scores
        else:
            result['user_pos_emb'] = None
            result['user_neg_emb'] = None
            result['item_pos_emb'] = None
            result['fused_emb'] = sequential_emb
            result['fused_scores'] = self.output_projection(sequential_emb)

        return result

    def barlow_twins_loss(self, z_seq: torch.Tensor, z_graph: torch.Tensor) -> torch.Tensor:
        """
        Barlow Twins Alignment Loss for redundancy reduction.

        Computes cross-correlation matrix between sequential and graph
        embeddings and penalizes deviation from identity.

        The loss has two terms:
        - On-diagonal: Forces correlation of corresponding features to 1 (alignment)
        - Off-diagonal: Forces correlation of different features to 0 (redundancy reduction)

        Args:
            z_seq: Sequential embeddings from SASRec (batch_size, embedding_dim)
            z_graph: Graph embeddings from PoneGNN (batch_size, embedding_dim)

        Returns:
            Barlow Twins loss scalar
        """
        batch_size = z_seq.size(0)

        # Normalize along feature dimension (zero mean, unit variance)
        z_seq_norm = (z_seq - z_seq.mean(dim=0)) / (z_seq.std(dim=0) + 1e-8)
        z_graph_norm = (z_graph - z_graph.mean(dim=0)) / (z_graph.std(dim=0) + 1e-8)

        # Cross-correlation matrix: C_ij = correlation between seq feature i and graph feature j
        correlation = torch.mm(z_seq_norm.t(), z_graph_norm) / batch_size

        # Loss: diagonal to 1 (alignment), off-diagonal to 0 (redundancy reduction)
        diagonal = torch.diag(correlation)
        off_diagonal = correlation - torch.diag_embed(diagonal)

        # On-diagonal: (1 - corr_ii)^2 - want features to be similar across views
        on_diag_loss = ((1 - diagonal) ** 2).sum()

        # Off-diagonal: sum(corr_ij^2) for i != j - want to reduce redundancy
        off_diag_loss = (off_diagonal ** 2).sum()

        return on_diag_loss + self.barlow_weight * off_diag_loss

    def orthogonal_loss(self, pos_emb: torch.Tensor, neg_emb: torch.Tensor) -> torch.Tensor:
        """
        Orthogonal loss to decorrelate positive and negative embeddings.

        Encourages positive and negative embeddings to be orthogonal
        (uncorrelated), enabling them to learn distinct representations
        for positive vs negative feedback patterns.

        Args:
            pos_emb: Positive embeddings from PoneGNN (batch_size, embedding_dim)
            neg_emb: Negative embeddings from PoneGNN (batch_size, embedding_dim)

        Returns:
            Orthogonal regularization loss scalar
        """
        # L2 normalize
        pos_norm = F.normalize(pos_emb, dim=1)
        neg_norm = F.normalize(neg_emb, dim=1)

        # Cosine similarity - should be close to 0 (orthogonal)
        similarity = (pos_norm * neg_norm).sum(dim=1)

        # Penalize non-orthogonality: minimize squared cosine similarity
        return (similarity ** 2).mean()

    def dual_feedback_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
        ratings: torch.Tensor = None,
    ) -> tuple:
        """
        Dual-Feedback Loss: Separate BPR for positive and negative feedback.

        Positive feedback (rating > 3.5): Learn to rank positive items higher
        Negative feedback (rating < 3.5): Learn to rank negative items lower

        This implements the key insight from Pone-GNN: negative feedback should
        actively push away unwanted items, not just rank them lower.

        Args:
            users: User indices (batch_size,)
            pos_items: Positive item indices (batch_size,)
            neg_items: Negative sample indices (batch_size,)
            pos_emb: Positive embeddings from PoneGNN (num_users + num_items, dim)
            neg_emb: Negative embeddings from PoneGNN (num_users + num_items, dim)
            ratings: Optional ratings for weighting

        Returns:
            Tuple of (positive_bpr_loss, negative_bpr_loss)
        """
        # Get embeddings for users and items
        u_pos = pos_emb[users]
        u_neg = neg_emb[users]
        i_pos = pos_emb[self.num_users + pos_items]
        n_pos = pos_emb[self.num_users + neg_items]
        i_neg = neg_emb[self.num_users + pos_items]
        n_neg = neg_emb[self.num_users + neg_items]

        # Positive BPR: maximize u_pos · i_pos - u_pos · n_pos
        # Learn to rank positively-rated items higher than negative samples
        pos_scores = (u_pos * i_pos).sum(dim=1)
        neg_scores = (u_pos * n_pos).sum(dim=1)
        pos_bpr = -F.logsigmoid(pos_scores - neg_scores).mean()

        # Negative BPR: maximize u_neg · n_neg - u_neg · i_neg
        # Learn to push away negatively-rated items
        neg_scores_neg = (u_neg * n_neg).sum(dim=1)
        pos_scores_neg = (u_neg * i_neg).sum(dim=1)
        neg_bpr = -F.logsigmoid(neg_scores_neg - pos_scores_neg).mean()

        return pos_bpr, neg_bpr

    def compute_joint_loss(
        self,
        item_sequences: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        negative_samples: torch.Tensor,
        epoch: int,
        ratings: torch.Tensor = None,
        apply_contrastive: bool = True,
    ) -> dict:
        """
        Compute CREATE++ joint loss with all five components.

        Loss components:
        1. Sequential Loss (SASRec Cross-Entropy) - learns temporal patterns
        2. Dual-Feedback Loss (Positive/Negative BPR) - learns from ratings
        3. Barlow Twins Alignment Loss - redundancy reduction between encoders
        4. Orthogonal Loss - decorrelates pos/neg embeddings
        5. Contrastive Loss - InfoNCE-style cross-space alignment (every 10 epochs)

        Args:
            item_sequences: Item sequences (batch_size, seq_len)
            mask: Boolean mask (batch_size, seq_len)
            labels: Target item labels (batch_size,)
            pos_edge_index: Positive edge indices (2, num_pos_edges)
            neg_edge_index: Negative edge indices (2, num_neg_edges)
            negative_samples: Negative samples for BPR (batch_size,)
            epoch: Current epoch (for contrastive loss triggering)
            ratings: Optional ratings for dual-feedback weighting
            apply_contrastive: Whether to apply contrastive loss

        Returns:
            Dictionary of loss components:
            - total_loss: Combined weighted loss
            - sequential_loss: SASRec cross-entropy
            - fused_loss: Fused prediction cross-entropy
            - dual_feedback_loss: Combined positive + negative BPR
            - pos_bpr_loss: Positive BPR loss
            - neg_bpr_loss: Negative BPR loss
            - barlow_loss: Barlow Twins alignment loss
            - orthogonal_loss: Orthogonal regularization loss
            - contrastive_loss: InfoNCE contrastive loss
        """
        users = item_sequences[:, 0]

        # Ensure negative samples have correct shape
        if negative_samples.dim() == 1:
            negative_samples = negative_samples.unsqueeze(1)
        neg_items = negative_samples.squeeze(1)

        # Forward pass
        output = self(
            item_sequences, mask,
            pos_edge_index, neg_edge_index,
            training_graph=True,
        )

        sequential_emb = output['sequential_emb']
        user_pos_emb = output['user_pos_emb']
        user_neg_emb = output['user_neg_emb']

        # 1. Sequential Loss (SASRec Cross-Entropy)
        # Learns temporal patterns in user behavior sequences
        seq_scores = output['sequential_scores']
        seq_loss = F.cross_entropy(seq_scores, labels)

        # 2. Dual-Feedback Loss (Positive/Negative BPR)
        # Learns separately from positive and negative interactions
        pos_bpr_loss, neg_bpr_loss = self.dual_feedback_loss(
            users=users,
            pos_items=labels,
            neg_items=neg_items,
            pos_emb=self.graph_encoder.pos_emb,
            neg_emb=self.graph_encoder.neg_emb,
            ratings=ratings,
        )
        dual_feedback_loss = self.dual_feedback_weight * (pos_bpr_loss + neg_bpr_loss)

        # 3. Barlow Twins Alignment Loss
        # Reduces redundancy between sequential and graph encoders
        barlow_loss = self.barlow_twins_loss(sequential_emb, user_pos_emb)

        # 4. Orthogonal Loss
        # Decorrelates positive and negative embeddings
        ortho_loss = self.orthogonal_loss(user_pos_emb, user_neg_emb)

        # 5. Contrastive Loss (InfoNCE-style)
        # Applied every N epochs as per Pone-GNN design
        # Aligns positive and negative embedding spaces
        contrastive_loss = torch.tensor(0.0, device=sequential_emb.device)
        if apply_contrastive and (epoch % self.contrastive_interval == 1 or epoch == 1):
            contrastive_loss = self.graph_encoder.compute_contrastive_loss(
                users=users,
                pos_items=labels,
                neg_items=neg_items,
            )

        # Fused prediction loss
        fused_scores = output['fused_scores']
        fused_loss = F.cross_entropy(fused_scores, labels)

        # Total loss: sum of all components with their weights
        total_loss = (
            seq_loss +
            fused_loss +
            dual_feedback_loss +
            self.barlow_weight * barlow_loss +
            self.orthogonal_weight * ortho_loss +
            contrastive_loss
        )

        return {
            'total_loss': total_loss,
            'sequential_loss': seq_loss,
            'fused_loss': fused_loss,
            'dual_feedback_loss': dual_feedback_loss,
            'pos_bpr_loss': pos_bpr_loss,
            'neg_bpr_loss': neg_bpr_loss,
            'barlow_loss': barlow_loss,
            'orthogonal_loss': ortho_loss,
            'contrastive_loss': contrastive_loss,
        }

    def predict(
        self,
        item_sequences: torch.Tensor,
        mask: torch.Tensor,
        pos_edge_index: torch.Tensor = None,
        neg_edge_index: torch.Tensor = None,
        top_k: int = 10,
    ) -> torch.Tensor:
        """
        Generate top-k recommendations.

        Args:
            item_sequences: Item sequences (batch_size, seq_len)
            mask: Boolean mask (batch_size, seq_len)
            pos_edge_index: Positive edge indices (optional for inference)
            neg_edge_index: Negative edge indices (optional for inference)
            top_k: Number of recommendations to return

        Returns:
            Top-k item indices (batch_size, top_k)
        """
        output = self(
            item_sequences, mask,
            pos_edge_index, neg_edge_index,
            training_graph=(pos_edge_index is not None),
        )

        scores = output['fused_scores']
        scores[:, 0] = -float('inf')  # Mask padding item

        _, indices = torch.topk(scores, k=top_k, dim=-1)
        return indices

    @torch.no_grad()
    def get_all_item_scores(self, user_embedding: torch.Tensor) -> torch.Tensor:
        """
        Get recommendation scores for all items.

        Args:
            user_embedding: User embedding (batch_size, embedding_dim)

        Returns:
            Scores for all items (batch_size, num_items + 1)
        """
        return self.output_projection(user_embedding)
