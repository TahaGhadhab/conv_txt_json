# Commandes vocales de drone — couche d'interprétation (prototype simulé)

```text
Transcription finale (arabe, dialecte tunisien) → normalisation → intention
→ validation → message JSON → répartiteur → adaptateur SIMULÉ
```

> **Simulation uniquement.** Aucun drone, autopilote, réseau ou modèle physique de vol n'est
> utilisé. Le simulateur enregistre des appels en mémoire ; ce n'est pas un simulateur de vol.
> Les transcriptions des exemples sont **simulées** : le modèle de reconnaissance réel n'est
> pas présent dans ce dépôt et n'a pas été intégré.

Le JSON décrit une commande, il n'exécute rien. Un message `recognized` signifie « intention
reconnue », pas « exécutée » : l'exécution (simulée) est décrite par le résultat du répartiteur.

## Prérequis

- Python ≥ 3.10 (seule la version 3.14.6 a été réellement testée). Aucune dépendance d'exécution :
  bibliothèque standard uniquement.
- `pytest` (≥ 7) pour les tests : `python -m pip install pytest`.

Toutes les commandes se lancent depuis la racine du dépôt.

## Démarrage rapide

```bash
# 1. Vérifier la configuration (vocabulaire + bornes de simulation)
python -m drone_voice check-config

# 2. Une transcription finale → JSON → résultat simulé
python -m drone_voice demo "تقدّم"
python -m drone_voice demo "ما تتقدّمش"            # négation : rejetée, aucun mouvement
python -m drone_voice demo "تقدّم" --partial       # partielle : rien n'est produit ni exécuté
python -m drone_voice demo "اطلع" --now 2026-01-01T12:00:00Z --utterance-id u1 --save-json out/m.json

# 3. Plusieurs entrées (JSON Lines), avec mémoire persistante des commandes déjà traitées
python -m drone_voice batch examples/transcripts_simulated.jsonl --out out/run1.jsonl
#    relancer le même fichier : toutes les commandes finales sont rejetées en DUPLICATE
python -m drone_voice batch examples/transcripts_simulated.jsonl --out out/run2.jsonl

# 4. Rejouer un message JSON enregistré (horloge réelle) : un ancien fichier est EXPIRED
python -m drone_voice dispatch examples/message_recognized.json

# 5. Tests
python -m pytest
```

Sous PowerShell, les mêmes commandes fonctionnent. Les dossiers des fichiers de sortie sont créés
au besoin (`out/`, `state/` sont ignorés par git). La sortie console est forcée en UTF-8.

Le mode `demo` utilise une horloge simulée : après l'envoi, il avance le temps simulé jusqu'à la
fin du mouvement pour montrer l'arrêt automatique, sans attendre réellement.

### Mode `batch`

Chaque ligne du fichier d'entrée est un objet JSON :

| Champ | Obligatoire | Description |
|---|---|---|
| `utterance_id` | oui | identifiant de la prise de parole (identique pour ses partielles et retransmissions) |
| `text` | oui | texte transcrit (tout type JSON accepté en entrée ; un non-texte est rejeté) |
| `is_final` | oui | `true` pour la transcription finale ; les partielles ne sont jamais exécutées |
| `t` | non | secondes simulées depuis le début, non décroissantes (défaut : valeur précédente) |
| `source`, `confidence`, `comment` | non | provenance, confiance fournie par le modèle, commentaire |

Garanties : le fichier entier est validé avant toute action ; le fichier de sortie n'est jamais
écrasé (erreur, code 2) ; les identifiants consommés sont enregistrés dans `--state`
(défaut `state/processed_command_ids.jsonl`), donc relancer le même fichier ne réexécute rien.

## Organisation

| Chemin | Rôle |
|---|---|
| `drone_voice/normalization.py` | normalisation déterministe documentée (v1) |
| `drone_voice/vocabulary.py` | vocabulaire configurable, correspondance exacte, refus des ambiguïtés |
| `drone_voice/interpreter.py` | transcription finale → message JSON validé |
| `drone_voice/contract.py` | contrat versionné et validateur unique |
| `drone_voice/dispatcher.py` | validation à la réception, déduplication, délais, STOP, appels adaptateur |
| `drone_voice/controller.py` | interface `DroneController` + `SimulatedController` |
| `drone_voice/transcription.py` | point d'entrée des événements du modèle (à brancher) |
| `drone_voice/dedup.py`, `decision_log.py` | identifiants consommés, journal de décisions |
| `drone_voice/cli.py` | démonstration en ligne de commande |
| `config/vocabulary.json` | vocabulaire **provisoire** |
| `config/settings.json` | validité des messages et profils de mouvement **de simulation** |
| `docs/CONTRACT.md` | contrat JSON et politiques du répartiteur |
| `docs/INTEGRATION.md` | où brancher le modèle et un futur adaptateur de drone |
| `examples/` | fixtures simulées, messages et sorties d'exemple |
| `QA_HANDOFF.md` | note de passage au QA |

## Vocabulaire

`config/vocabulary.json` contient les sept expressions de démarrage fournies. **Elles ne
constituent pas un lexique tunisien certifié** : elles doivent être validées avec les utilisateurs
et confrontées aux sorties réelles du modèle (orthographe, diacritiques, hamza…) avant tout usage.

- Correspondance **exacte** sur la phrase complète normalisée ; aucune recherche de sous-chaîne,
  aucun rapprochement approximatif, aucun LLM, aucune segmentation. `تقدّم و اطلع`, `قلتلك تقدّم`
  et `ما تتقدّمش` ne produisent donc aucun mouvement.
- Le chargement échoue si deux expressions identiques après normalisation désignent des intentions
  différentes, si une intention n'a aucune expression, ou si une expression est vide.
- Ajouter une variante = ajouter une entrée `{"expression": ..., "intent": ...}`.

Règles de normalisation (v1, `drone_voice/normalization.py`) : NFC ; suppression des harakat
(U+064B–U+0652) et de l'alif suscrit (U+0670) ; suppression du tatweel (U+0640) ; suppression des
marques invisibles LRM/RLM/ALM/BOM ; espaces fusionnés et rognés ; suppression de `. ! ? ؟` en fin
de phrase seulement ; NFC final (fonction idempotente). Les lettres (أ/إ/آ/ا, ة/ه, ى/ي) ne sont pas
unifiées et la ponctuation interne est conservée. Le texte original est toujours conservé dans
`recognized_text`.

## Journaux

Désactivés par défaut. `--decision-log FICHIER` ajoute une ligne JSON par décision
(`partial_ignored`, `interpreted`, `dispatched`), reliées par `utterance_id` et `command_id`.
`--no-log-text` retire tout texte transcrit du journal (`text_redacted: true`). Aucun audio ni
secret n'est enregistré. Le fichier d'identifiants traités (`state/`) ne contient que des
identifiants et des dates.

## Limites

Voir `QA_HANDOFF.md` et `docs/INTEGRATION.md`. En résumé : modèle réel non intégré (entrées
simulées), aucun adaptateur de drone réel, paramètres de mouvement non validés pour le vol,
garantie « au plus une fois » limitée au fichier d'état utilisé, aucune garantie « exactement une
fois » vis-à-vis d'un futur drone.
