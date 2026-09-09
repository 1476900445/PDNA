import torch

from pdna.bayesian import BayesianFeatureRecognizer, ConditionalProbabilityTables
from pdna.expo import ExpressivePolicyLayer
from pdna.fql import FQLAgent, InverseUtilityMapper, Outcome, build_fql_state, price_utility
from pdna.setformer import SeTformer, SeTformerBatch
from pdna.setformer.transport import SinkhornSolver


def test_bayesian_output_shapes_and_normalization():
    model = BayesianFeatureRecognizer(256)
    result = model(torch.randn(3, 256), torch.zeros(3, 128),
                   ConditionalProbabilityTables.uniform())
    assert result.personality_probs.shape == (3, 5)
    assert result.intent_probs.shape == (3, 23)
    assert result.strategy_probs.shape == (3, 10)
    assert torch.allclose(result.intent_probs.sum(-1), torch.ones(3), atol=1e-5)
    assert ((0 <= result.addon_sensitivity) & (result.addon_sensitivity <= 1)).all()


def test_expo_two_candidate_selection():
    layer = ExpressivePolicyLayer(40, 23, 24, [64, 64])
    probabilities, action, candidates = layer(torch.randn(4, 40), deterministic=True)
    assert candidates.q_values.shape == (4, 2)
    assert probabilities.shape == (4, 23)
    assert action.shape == (4,)


def test_fql_shapes_and_state_equation():
    state = build_fql_state([0.2, 0.3], [0.9, 0.8], 5, 10)
    assert torch.allclose(state, torch.tensor([0.0, 0.0, 0.2, 0.9, 0.3, 0.8, 0.5]))
    model = FQLAgent(hidden_dims=[32, 32], flow_steps=3)
    assert model.act(state[None]).shape == (1, 1)
    assert price_utility(80, 60, 100) == 0.5


def test_inverse_utility_prefers_opponent():
    outcomes = [Outcome({"id": "a"}), Outcome({"id": "b"})]
    own = {"a": 0.80, "b": 0.805}
    opp = {"a": 0.4, "b": 0.7}
    mapper = InverseUtilityMapper(outcomes, lambda x: own[x.values["id"]],
                                  lambda x: opp[x.values["id"]], 0.01)
    assert mapper.map(0.8).values["id"] == "b"


def test_sinkhorn_marginals():
    plan = SinkhornSolver(0.1, 100)(torch.rand(2, 4, 7))
    assert torch.allclose(plan.sum(2), torch.full((2, 4), 0.25), atol=1e-3)
    assert torch.allclose(plan.sum(1), torch.full((2, 7), 1 / 7), atol=1e-3)


def test_setformer_joint_loss_is_finite():
    model = SeTformer(
        terminology_input_dim=49, interaction_input_dim=40, history_input_dim=16,
        dimension=32, reference_points=8, nystrom_landmarks=4,
        vocabulary_size=100, decoder_layers=2, decoder_heads=4, max_length=12,
    )
    batch = SeTformerBatch(
        terminology=torch.randn(2, 49),
        interaction=torch.randn(2, 40),
        history=torch.randn(2, 5, 16),
        history_mask=torch.ones(2, 5, dtype=torch.bool),
        decoder_input_ids=torch.randint(0, 100, (2, 6)),
        target_ids=torch.randint(0, 100, (2, 6)),
        goal_embedding=torch.randn(2, 32),
        target_intent=torch.tensor([1, 2]),
        target_politeness=torch.tensor([0.8, 0.7]),
    )
    losses = model.losses(batch)
    assert torch.isfinite(losses.total)
