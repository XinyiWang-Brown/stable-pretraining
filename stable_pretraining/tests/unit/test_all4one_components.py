"""Reference parity and state/gradient contracts for All4One components."""

import copy
import math
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from stable_pretraining.utils import all4one
from stable_pretraining.utils.all4one import (
    All4OneAttention,
    All4OneQueue,
    build_all4one_predictor,
    build_all4one_projector,
)

pytestmark = pytest.mark.unit


def test_queue_starts_full_with_normalized_random_vectors():
    torch.manual_seed(13)
    expected = F.normalize(torch.randn(11, 8), dim=-1)
    torch.manual_seed(13)
    queue = All4OneQueue(8, queue_size=11)
    torch.testing.assert_close(queue.queue.get(), expected)
    assert queue.queue.pointer.item() == 0
    assert queue.queue.filled.item()
    assert not list(queue.parameters())


def test_queue_returns_sorted_cosine_neighbours_without_mutation():
    queue = All4OneQueue(2, queue_size=5, num_neighbours=3)
    support = torch.tensor([[1.0, 0], [0, 1], [-1, 0], [1, 1], [-1, -1]])
    queue.enqueue(support)
    state = copy.deepcopy(queue.state_dict())
    query = torch.tensor([[8.0, 1.0], [-1.0, 3.0]], requires_grad=True)
    result = queue(query)
    assert result["indices"].tolist() == [[0, 3, 1], [1, 3, 2]]
    expected = F.normalize(support, dim=-1)[result["indices"]]
    torch.testing.assert_close(result["neighbours"], expected)
    assert not result["neighbours"].requires_grad
    for name, value in queue.state_dict().items():
        torch.testing.assert_close(value, state[name])
    queue.enqueue(torch.randn(5, 2))
    torch.testing.assert_close(result["neighbours"], expected)


@pytest.mark.parametrize("sizes", [(2, 2, 3), (5, 1, 4), (7, 2), (0, 3)])
def test_queue_reuses_fifo_wraparound_and_oversized_batch_handling(sizes):
    queue = All4OneQueue(3, queue_size=5, num_neighbours=2)
    history = list(queue.queue.get().clone())
    for size in sizes:
        batch = torch.randn(size, 3, requires_grad=True)
        history.extend(F.normalize(batch.detach(), dim=-1))
        history = history[-5:]
        queue.enqueue(batch)
        pointer = int(queue.queue.pointer)
        chronological = torch.roll(queue.queue.get(), shifts=-pointer, dims=0)
        torch.testing.assert_close(chronological, torch.stack(history))
        assert not queue.queue.out.requires_grad


def test_enqueue_gathers_detached_normalized_features_in_rank_order(monkeypatch):
    local = torch.tensor([[3.0, 4.0], [0.0, 2.0]], requires_grad=True)
    remote = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])
    gather = Mock(side_effect=lambda x: [x, remote])
    monkeypatch.setattr(all4one, "all_gather", gather)
    queue = All4OneQueue(2, queue_size=5, num_neighbours=2)
    queue.enqueue(local)
    passed = gather.call_args.args[0]
    assert not passed.requires_grad
    torch.testing.assert_close(passed, F.normalize(local.detach(), dim=-1))
    torch.testing.assert_close(queue.queue.get()[:4], torch.cat((passed, remote)))
    assert queue.queue.pointer.item() == 4


def test_nested_queue_checkpoint_restores_lookup_and_next_insertion():
    original = nn.ModuleDict({"support": All4OneQueue(4, 7, 3)})
    original["support"].enqueue(torch.randn(3, 4))
    restored = nn.ModuleDict({"support": All4OneQueue(4, 7, 3)})
    restored.load_state_dict(copy.deepcopy(original.state_dict()), strict=True)
    query = torch.randn(2, 4)
    for name, value in original["support"](query).items():
        torch.testing.assert_close(restored["support"](query)[name], value)
    next_batch = torch.randn(5, 4)
    original["support"].enqueue(next_batch)
    restored["support"].enqueue(next_batch)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_queue_dtype_and_autocast_contract(dtype):
    queue = All4OneQueue(8, 11, 3)
    query = torch.randn(3, 8).to(dtype)
    baseline = queue(query)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = queue(query)
        queue.enqueue(query)
    assert result["neighbours"].dtype == dtype
    torch.testing.assert_close(result["indices"], baseline["indices"])
    torch.testing.assert_close(queue.queue.out[:3], F.normalize(query.float(), dim=-1))
    assert queue.queue.out.dtype == torch.float32


def _reference_position(k, dim):
    denominator = 2 * math.ceil(k / 2)
    return torch.tensor(
        [
            [
                (math.sin if channel % 2 == 0 else math.cos)(
                    position / 10000 ** (2 * (channel // 2) / denominator)
                )
                for position in range(dim)
            ]
            for channel in range(k)
        ],
        dtype=torch.float32,
    ).unsqueeze(0)


@pytest.mark.parametrize("k", [1, 4, 5])
@pytest.mark.parametrize("shifted", [False, True])
def test_attention_matches_reference_encoding_shift_and_first_token(k, shifted):
    attention = All4OneAttention(8, k, num_heads=2, num_layers=2, dropout=0)
    reference = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            8, 2, dim_feedforward=16, dropout=0, batch_first=True
        ),
        num_layers=2,
    )
    reference.load_state_dict(attention.encoder.state_dict())
    neighbours = torch.randn(3, k, 8)
    prediction = torch.randn(3, 8, requires_grad=True)
    reference_prediction = prediction.detach().clone().requires_grad_()
    tokens = neighbours
    if shifted:
        tokens = torch.cat((reference_prediction[:, None], neighbours), dim=1)[:, :k]
    expected = reference(tokens + _reference_position(k, 8))[:, 0]
    actual = attention(neighbours, prediction if shifted else None)
    torch.testing.assert_close(actual, expected)
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    for parameter, ref_parameter in zip(
        attention.encoder.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(
            parameter.grad, ref_parameter.grad, rtol=1e-4, atol=2e-6
        )
    if shifted:
        torch.testing.assert_close(prediction.grad, reference_prediction.grad)
        assert prediction.grad.abs().sum() > 0
    else:
        assert prediction.grad is None


def test_attention_defaults_and_checkpoint_round_trip():
    attention = All4OneAttention()
    assert len(attention.encoder.layers) == 3
    layer = attention.encoder.layers[0]
    assert layer.self_attn.num_heads == 8
    assert layer.linear1.out_features == 512
    assert layer.dropout.p == 0.1
    assert not layer.norm_first
    assert attention.encoder.norm is None
    restored = All4OneAttention()
    restored.load_state_dict(attention.state_dict(), strict=True)
    attention.eval()
    restored.eval()
    inputs = torch.randn(2, 5, 256)
    torch.testing.assert_close(attention(inputs), restored(inputs))


def test_both_attention_branches_train_after_queue_is_updated():
    queue = All4OneQueue(8, 7, 5)
    momentum = torch.randn(3, 8, requires_grad=True)
    neighbours = queue(momentum)["neighbours"]
    attention = All4OneAttention(8, num_heads=2, num_layers=1, dropout=0)
    prediction = torch.randn(3, 8, requires_grad=True)
    centroid = attention(neighbours)
    shifted = attention(neighbours, prediction)
    centroid.retain_grad()
    shifted.retain_grad()
    queue.enqueue(momentum)
    (centroid * shifted).sum().backward()
    assert momentum.grad is None
    assert prediction.grad.abs().sum() > 0
    assert centroid.grad.abs().sum() > 0
    assert shifted.grad.abs().sum() > 0
    assert attention.encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["projector", "predictor"])
def test_mlp_builders_match_reference_outputs_gradients_and_batchnorm(kind):
    if kind == "projector":
        actual = build_all4one_projector(6, 12, 8)
        expected = nn.Sequential(
            nn.Linear(6, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Linear(12, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.BatchNorm1d(8),
        )
        inputs = torch.randn(4, 6)
    else:
        actual = build_all4one_predictor(8, 12)
        expected = nn.Sequential(
            nn.Linear(8, 12), nn.BatchNorm1d(12), nn.ReLU(), nn.Linear(12, 8)
        )
        inputs = torch.randn(4, 8)
    with torch.no_grad():
        for parameter, ref_parameter in zip(
            actual.parameters(), expected.parameters(), strict=True
        ):
            ref_parameter.copy_(parameter)
    result = actual(inputs)
    reference = expected(inputs)
    torch.testing.assert_close(result, reference)
    weight = torch.randn_like(result)
    (result * weight).sum().backward()
    (reference * weight).sum().backward()
    for parameter, ref_parameter in zip(
        actual.parameters(), expected.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter.grad, ref_parameter.grad)
    for buffer, ref_buffer in zip(actual.buffers(), expected.buffers(), strict=True):
        torch.testing.assert_close(buffer, ref_buffer)


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (All4OneQueue, {"projection_dim": 0}),
        (All4OneQueue, {"queue_size": -1}),
        (All4OneQueue, {"num_neighbours": 0}),
        (All4OneQueue, {"queue_size": 3, "num_neighbours": 4}),
        (All4OneQueue, {"queue_size": 1.5}),
        (All4OneQueue, {"projection_dim": True}),
        (All4OneAttention, {"projection_dim": 0}),
        (All4OneAttention, {"num_neighbours": 0}),
        (All4OneAttention, {"num_heads": 0}),
        (All4OneAttention, {"num_layers": 0}),
        (All4OneAttention, {"projection_dim": 10, "num_heads": 3}),
        (All4OneAttention, {"dropout": -0.1}),
        (All4OneAttention, {"dropout": 1.1}),
        (build_all4one_projector, {"input_dim": 0}),
        (build_all4one_projector, {"input_dim": 4, "hidden_dim": 0}),
        (build_all4one_predictor, {"projection_dim": 0}),
        (build_all4one_predictor, {"hidden_dim": -1}),
    ],
)
def test_components_reject_invalid_dimensions_and_configuration(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


@pytest.mark.parametrize(
    "inputs", [torch.ones(3), torch.ones(2, 3), torch.ones(2, 4, dtype=torch.long)]
)
def test_queue_rejects_invalid_inputs(inputs):
    queue = All4OneQueue(4, 7)
    for call in (queue, queue.enqueue):
        with pytest.raises(ValueError, match="projections"):
            call(inputs)


@pytest.mark.parametrize("shape", [(2, 8), (2, 4, 8), (2, 5, 4)])
def test_attention_rejects_wrong_neighbour_shapes(shape):
    attention = All4OneAttention(8, num_heads=2, num_layers=1)
    with pytest.raises(ValueError, match="neighbours"):
        attention(torch.randn(shape))


def test_attention_rejects_wrong_prediction_shape():
    attention = All4OneAttention(8, num_heads=2, num_layers=1)
    with pytest.raises(ValueError, match="prediction"):
        attention(torch.randn(2, 5, 8), torch.randn(2, 1, 8))
