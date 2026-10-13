import pytest
import torch

from delphi_demo import tokeny, trening

LOG = """number of parameters: 2.24M
iter 0: loss 6891.8218, time 9316.82ms
iter 50: loss 6943.6167, time 335.74ms
step 100: train loss 3396.5364, val loss 3414.7507
saving checkpoint to /x
iter 100: loss 6554.7964, time 15196.12ms
"""


def test_polecenie_treningu(tmp_path):
    p = trening.polecenie_treningu(tmp_path, kroki=20, urzadzenie="cpu", co_ile_ewaluacja=10,
                                   partie_ewaluacji=2)
    assert p[1:4] == ["-u", "train.py", "config/train_delphi_demo.py"]
    assert "--device=cpu" in p and "--max_iters=20" in p and "--lr_decay_iters=20" in p
    assert "--warmup_iters=2" in p and f"--out_dir={tmp_path.resolve()}" in p


def test_polecenie_pelne_jak_config_autorow(tmp_path):
    p = trening.polecenie_treningu(tmp_path)
    assert "--max_iters=5000" in p and "--warmup_iters=500" in p and "--eval_interval=250" in p


def test_polecenie_szybka_sciezka_2500(tmp_path):
    p = trening.polecenie_treningu(tmp_path, kroki=2500)
    assert "--max_iters=2500" in p and "--lr_decay_iters=2500" in p and "--warmup_iters=250" in p


def test_parsuj_log():
    kroki, ewal = trening.parsuj_log(LOG)
    assert kroki["iter"].tolist() == [0, 50, 100]
    assert ewal.iloc[0].to_dict() == {"krok": 100, "trening": 3396.5364, "walidacja": 3414.7507}


def test_odchudz_i_zaladuj(maly_model, tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    args = dict(block_size=96, vocab_size=tokeny.LICZBA_TOKENOW, n_layer=1, n_head=2, n_embd=16,
                dropout=0.0, t_min=0.1, bias=False, mask_ties=True, ignore_tokens=list(tokeny.IGNOROWANE))
    torch.save({"model": maly_model.state_dict(), "optimizer": {"duzy": torch.zeros(1000)},
                "model_args": args, "iter_num": 20, "best_val_loss": 70.5, "config": {}}, ckpt)
    cel = trening.odchudz_checkpoint(ckpt, tmp_path / "maly.pt")
    model = trening.zaladuj_model(cel)
    x = torch.tensor([[tokeny.KOBIETA, 215, tokeny.BRAK_ZDARZENIA]])
    a = torch.tensor([[0.0, 18262.5, 21915.0]])
    assert torch.allclose(model(x, a)[0], maly_model(x, a)[0])
    assert "optimizer" not in torch.load(cel, weights_only=True)
    assert trening.liczba_parametrow(model) == trening.liczba_parametrow(maly_model)


def test_brak_modelu_mowi_co_zrobic(tmp_path):
    with pytest.raises(FileNotFoundError, match="przygotuj_demo"):
        trening.zaladuj_model(tmp_path / "nie_ma.pt")


@pytest.mark.wolny
def test_krotki_prawdziwy_trening(tmp_path):
    ckpt = trening.trenuj(tmp_path, kroki=20, urzadzenie="cpu", co_ile_ewaluacja=10,
                          partie_ewaluacji=2, co_ile_log=5)
    assert ckpt.exists()
    kroki, ewal = trening.parsuj_log((tmp_path / "trening.log").read_text())
    assert len(kroki) >= 4 and len(ewal) == 2
    model = trening.zaladuj_model(trening.odchudz_checkpoint(ckpt, tmp_path / "m.pt"))
    assert trening.liczba_parametrow(model) == 2_243_400
