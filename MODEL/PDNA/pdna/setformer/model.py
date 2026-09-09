from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoders import GatedFusion, HistoryBiLSTM, InteractionFeatureEncoder, TerminologyCNN
from .transport import SelfOptimalTransport


@dataclass
class SeTformerBatch:
    terminology: Tensor
    interaction: Tensor
    history: Tensor
    history_mask: Tensor
    decoder_input_ids: Tensor
    target_ids: Tensor
    goal_embedding: Tensor
    target_intent: Tensor
    target_politeness: Tensor


@dataclass
class SeTformerLosses:
    total: Tensor
    generation: Tensor
    goal_alignment: Tensor
    intent_matching: Tensor
    regularization: Tensor
    politeness: Tensor


class CausalTransformerGenerator(nn.Module):
    def __init__(self, vocabulary_size: int, dimension: int, layers: int = 6,
                 heads: int = 8, max_length: int = 200):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, dimension)
        self.position = nn.Embedding(max_length, dimension)
        layer = nn.TransformerDecoderLayer(dimension, heads, 4 * dimension,
                                           dropout=0.1, activation="gelu", batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.output = nn.Linear(dimension, vocabulary_size, bias=False)
        self.output.weight = self.embedding.weight
        self.max_length = max_length

    def forward(self, token_ids: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        length = token_ids.shape[1]
        positions = torch.arange(length, device=token_ids.device).unsqueeze(0)
        hidden = self.embedding(token_ids) + self.position(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(length, device=token_ids.device)
        hidden = self.decoder(hidden, memory, tgt_mask=causal_mask)
        return self.output(hidden), hidden


class SeTformer(nn.Module):
    """Equations (18)-(20): multimodal encoding, SeT alignment, constrained decoder."""

    def __init__(self, terminology_input_dim: int = 49, interaction_input_dim: int = 40,
                 history_input_dim: int = 256, dimension: int = 128, reference_points: int = 64,
                 nystrom_landmarks: int = 32, rbf_gamma: float | None = None,
                 sinkhorn_epsilon: float = 0.1, sinkhorn_iterations: int = 50,
                 vocabulary_size: int = 32000, decoder_layers: int = 6, decoder_heads: int = 8,
                 max_length: int = 200, importance_threshold: float = 0.95,
                 loss_weights: dict[str, float] | None = None):
        super().__init__()
        self.terminology_encoder = TerminologyCNN(terminology_input_dim, dimension)
        self.interaction_encoder = InteractionFeatureEncoder(interaction_input_dim, dimension)
        self.history_encoder = HistoryBiLSTM(history_input_dim, dimension)
        self.fusion = GatedFusion(3 * dimension, dimension)
        self.transport = SelfOptimalTransport(
            dimension, reference_points, nystrom_landmarks, rbf_gamma,
            sinkhorn_epsilon, sinkhorn_iterations
        )
        self.goal_match = nn.Bilinear(dimension, dimension, 1)
        self.politeness_head = nn.Sequential(nn.Linear(dimension, 1), nn.Sigmoid())
        self.intent_head = nn.Linear(dimension, 23)
        self.goal_projection = nn.Linear(dimension, dimension)
        self.generator = CausalTransformerGenerator(
            vocabulary_size, dimension, decoder_layers, decoder_heads, max_length
        )
        self.importance_threshold = importance_threshold
        self.weights = loss_weights or {
            "generation": 0.29, "goal": 0.15, "intent": 0.21,
            "regularization": 1e-5, "politeness": 0.20,
        }

    def encode(self, terminology: Tensor, interaction: Tensor, history: Tensor,
               history_mask: Tensor, goal_embedding: Tensor,
               token_politeness: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        et = self.terminology_encoder(terminology)
        ef = self.interaction_encoder(interaction)
        eh = self.history_encoder(history, history_mask)
        fused = self.fusion(et, ef, eh)
        source_tokens = torch.stack((et, ef, eh, fused), dim=1)
        aligned, plan, _ = self.transport(source_tokens)

        refs = self.transport.references
        goal = self.goal_projection(goal_embedding)
        match = torch.sigmoid(self.goal_match(
            refs.unsqueeze(0).expand(goal.shape[0], -1, -1),
            goal.unsqueeze(1).expand(-1, refs.shape[0], -1),
        ).squeeze(-1))
        importance = (plan * match.unsqueeze(1)).sum(-1)
        if token_politeness is not None:
            importance = importance * token_politeness
        # Keep the smallest ranked set whose cumulative normalized importance >= 0.95.
        normalized = importance / importance.sum(-1, keepdim=True).clamp_min(1e-12)
        order = normalized.argsort(-1, descending=True)
        cumulative = normalized.gather(1, order).cumsum(-1)
        sorted_scores = normalized.gather(1, order)
        keep_ranked = (cumulative - sorted_scores) < self.importance_threshold
        keep = torch.zeros_like(keep_ranked).scatter(1, order, keep_ranked)
        memory = aligned * keep.unsqueeze(-1)
        return memory, plan, fused

    def forward(self, batch: SeTformerBatch) -> tuple[Tensor, Tensor, Tensor]:
        memory, _, fused = self.encode(
            batch.terminology, batch.interaction, batch.history,
            batch.history_mask, batch.goal_embedding
        )
        logits, hidden = self.generator(batch.decoder_input_ids, memory)
        return logits, hidden, fused

    def losses(self, batch: SeTformerBatch) -> SeTformerLosses:
        logits, hidden, _fused = self(batch)
        generation = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch.target_ids.reshape(-1), ignore_index=-100
        )
        generated_embedding = hidden.mean(1)
        goal_alignment = torch.linalg.vector_norm(
            generated_embedding - self.goal_projection(batch.goal_embedding), dim=-1
        ).mean()
        intent_matching = F.cross_entropy(self.intent_head(generated_embedding), batch.target_intent)
        regularization = sum(parameter.square().sum() for parameter in self.parameters())
        generated_politeness = self.politeness_head(generated_embedding).squeeze(-1)
        politeness = torch.relu(batch.target_politeness - generated_politeness).mean()
        total = (
            self.weights["generation"] * generation
            + self.weights["goal"] * goal_alignment
            + self.weights["intent"] * intent_matching
            + self.weights["regularization"] * regularization
            + self.weights["politeness"] * politeness
        )
        return SeTformerLosses(total, generation, goal_alignment, intent_matching,
                               regularization, politeness)

    @torch.no_grad()
    def generate(self, memory: Tensor, bos_id: int, eos_id: int, temperature: float = 0.8,
                 max_length: int = 200) -> Tensor:
        ids = torch.full((memory.shape[0], 1), bos_id, dtype=torch.long, device=memory.device)
        finished = torch.zeros(memory.shape[0], dtype=torch.bool, device=memory.device)
        for _ in range(max_length - 1):
            logits, _ = self.generator(ids, memory)
            probabilities = torch.softmax(logits[:, -1] / temperature, -1)
            next_id = torch.multinomial(probabilities, 1)
            ids = torch.cat((ids, next_id), 1)
            finished |= next_id.squeeze(-1).eq(eos_id)
            if finished.all():
                break
        return ids
