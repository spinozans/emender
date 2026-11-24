import torch
from torch import nn
from experiments.expressivity_tasks.x45_matrix_emender.k15_phrase_qualification import PREFIX, j
from .k15_bookmarks import BookmarkSession
from .k15_phrase_recurrent import RecurrentReplay, PhraseK15


def test_chosen_generation_online_replay_and_reload():
    torch.manual_seed(3411)
    base = nn.Module(); base.embed = nn.Embedding(256,8); base.norm = nn.LayerNorm(8)
    base.blocks = nn.ModuleList([nn.Linear(8,8)]); base.head = nn.Linear(8,256)
    model = PhraseK15(base); clone = PhraseK15(base); clone.load_state_dict(model.state_dict(), strict=True)
    a,b,c = [BookmarkSession(j.EXE) for _ in range(3)]
    try:
        hard, online, restored = [RecurrentReplay(m,s,PREFIX) for m,s in ((model,a),(model,b),(clone,c))]
        for step in range(2):
            chosen = hard.predict(chosen=True); y=int(chosen.argmax()); hard.observe(y)
            assert y==b'AB'[step]
            p=online.predict(); q=restored.predict(); assert torch.equal(p,q) and abs(p.sum()-1)<1e-5 and p.min()>0
            online.observe(y); restored.observe(y)
    finally:
        for s in (a,b,c): s.close()
