from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .bayesian import BayesianFeatureRecognizer, ConditionalProbabilityTables, OpponentFeatures
from .config import PDNAConfig
from .decision import ConsistencyDecision, HighLowConsistencyResolver
from .expo import EXPOHierarchicalAgent, EXPOStateBuilder
from .fql import FQLAgent
from .setformer import SeTformer


@dataclass
class PDNATurnOutput:
    opponent: OpponentFeatures
    intent_probabilities: Tensor
    strategy_probabilities: Tensor
    utility_action: Tensor
    consistency: list[ConsistencyDecision]


class PDNA(nn.Module):
    """Five-module PDNA graph. Text decoding/correction are exposed as explicit final stages."""

    def __init__(self, config: PDNAConfig):
        super().__init__()
        b, e, f, s = config.raw.bayesian, config.raw.expo, config.raw.fql, config.raw.setformer
        self.config = config
        self.recognizer = BayesianFeatureRecognizer(
            b.utterance_vector_dim, b.history_dim, b.lexical_dim, list(b.hidden_dims),
            b.lambda_intent, b.lambda_strategy, b.posterior_eps
        )
        self.state_builder = EXPOStateBuilder(e.intent.state_dim, e.strategy.state_dim)
        self.expo = EXPOHierarchicalAgent(
            e.intent.state_dim, e.strategy.state_dim, list(e.hidden_dims), e.gamma, e.target_tau,
            e.edit_std_min, e.edit_std_max
        )
        self.fql = FQLAgent(
            f.state_dim, f.action_dim, list(f.hidden_dims), f.q_networks, f.gamma,
            f.target_tau, f.flow_steps, f.bc_coefficient, f.q_guidance_coefficient,
            f.action_min, f.action_max
        )
        self.setformer = SeTformer(
            s.terminology_input_dim, s.interaction_input_dim, s.history_input_dim,
            s.encoding_dim, s.reference_points, s.nystrom_landmarks, s.rbf_gamma,
            s.sinkhorn_epsilon, s.sinkhorn_iterations, s.vocabulary_size,
            s.decoder_layers, s.decoder_heads, s.max_generation_length,
            s.cumulative_importance_threshold, dict(s.loss_weights)
        )
        self.consistency_resolver = HighLowConsistencyResolver(
            {3: -1, 4: -1, 6: -1, 8: 1}  # Table 3 semantics; configurable in production.
        )

    def decide(self, utterance_vector: Tensor, previous_history: Tensor,
               cpt: ConditionalProbabilityTables, previous_opponent: OpponentFeatures,
               previous_our_intent: Tensor, previous_our_strategy: Tensor,
               fql_state: Tensor, previous_self_utility: Tensor) -> PDNATurnOutput:
        opponent = self.recognizer(utterance_vector, previous_history, cpt)
        intent_inputs = (
            previous_opponent.personality_probs, opponent.personality_probs,
            previous_opponent.intent_probs, previous_our_intent, opponent.intent_probs,
            previous_opponent.strategy_probs, previous_our_strategy, opponent.strategy_probs,
        )
        intent_state = self.state_builder.intent_state(*intent_inputs)

        def strategy_state(current_intent: Tensor) -> Tensor:
            return self.state_builder.strategy_state(intent_inputs, current_intent)

        high_level = self.expo(intent_state, strategy_state, deterministic=not self.training)
        utility = self.fql.act(fql_state)
        decisions = [
            self.consistency_resolver.resolve(
                int(high_level.intent[i]), int(high_level.strategy[i]),
                float(previous_self_utility[i].detach()), float(utility[i, 0].detach())
            )
            for i in range(utility.shape[0])
        ]
        return PDNATurnOutput(
            opponent, high_level.intent_probabilities, high_level.strategy_probabilities,
            utility, decisions
        )
