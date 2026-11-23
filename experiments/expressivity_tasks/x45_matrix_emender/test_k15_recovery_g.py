import torch
from .k15_recovery_g import draws


def test_sampling_matches_rng_state():
    torch.manual_seed(3407)
    origin=torch.get_rng_state()
    for _ in range(7):
        for logits in ([1.,0.], [0.,1.,-torch.inf,-torch.inf]):
            torch.distributions.Categorical(logits=torch.tensor(logits)).sample()
    expected=torch.get_rng_state(); torch.set_rng_state(origin); draws('cpu')
    assert torch.equal(expected,torch.get_rng_state())
