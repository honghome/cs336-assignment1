import torch

from cs336_basics.generate import generate_ids, sample_next_token, top_p_filtering


def test_top_p_filtering_keeps_minimal_probability_mass():
    probs = torch.tensor([0.50, 0.25, 0.15, 0.10])

    filtered = top_p_filtering(probs, top_p=0.70)

    assert torch.allclose(filtered, torch.tensor([2 / 3, 1 / 3, 0.0, 0.0], dtype=filtered.dtype))
    assert torch.allclose(filtered.sum(), torch.tensor(1.0))


def test_sample_next_token_temperature_zero_is_greedy():
    logits = torch.tensor([1.0, 3.0, 2.0])

    next_id = sample_next_token(logits, temperature=0.0)

    assert next_id.item() == 1


def test_generate_ids_uses_sliding_context_and_stops_at_eos():
    class DummyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_lengths = []

        def forward(self, input_ids):
            self.seen_lengths.append(input_ids.shape[1])
            batch_size, sequence_length = input_ids.shape
            logits = torch.zeros(batch_size, sequence_length, 5)
            next_ids = (input_ids[:, -1] + 1) % 5
            logits[:, -1, next_ids] = 10.0
            return logits

    model = DummyLM()

    generated = generate_ids(
        model=model,
        prompt_ids=[0, 1],
        max_new_tokens=4,
        context_length=3,
        eos_token_id=4,
        temperature=0.0,
    )

    assert generated.tolist() == [[0, 1, 2, 3, 4]]
    assert model.seen_lengths == [2, 3, 3]