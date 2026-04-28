import torch
from pathlib import Path
from miditok import REMI

tokenizer = REMI()

chemins_midi = list(Path("/Users/aiminemeddeb/Documents/Music_Generation_Transformers/data/maestro-v3.0.0").rglob("*.midi"))
maestro_tokens_list = []

print("Conversion des fichiers MIDI en Tokens...")
for chemin in chemins_midi:
    try:
        tokens = tokenizer(chemin)
        maestro_tokens_list.append(tokens[0].ids)
    except Exception as e:
        print(f"Fichier corrompu ignoré : {chemin}")

torch.save(maestro_tokens_list, "maestro_tokenized_complet.pt")
print("Base de données sauvegardée !")