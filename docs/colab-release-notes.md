# Colab demo artifacts

Szybki start notebooków w Google Colab **bez treningu** i **bez pobierania wag** (~700 MB).

| Plik | Zawartość | Rozmiar (orientacyjnie) |
|------|-----------|-------------------------|
| `delphi-pamiec-colab.tar.gz` | `delphi_syntetyczny.pt` + `pamiec/wyniki/*` (tryb `z_pamieci`) | ~9 MB |
| `usg-startowy-colab.tar.gz` | wycinek `.mha` (128 klatek) + `pamiec/*` dla tego przypadku | ~50 MB |

**Nie są w drzewie gita** (reguła: brak `*.pt` / `pamiec/` w repozytorium). Nadal potrzebny jest klon kodu autorów Delphi (MIT) oraz lekkie `pip` (PyTorch, SimpleITK…).

## Delphi — licencja checkpointu

Checkpoint wytrenowany na **danych syntetycznych** autorów Delphi (nie UK Biobank). Do użytku edukacyjnego przy sympozjum; nie jest wyrobem medycznym.

## USG — licencja nagrania

Wycinek z zbioru ACOUSLIC-AI (Zenodo 12697994), CC BY-NC-SA 4.0.