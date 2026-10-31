# Guide d'intégration

## 1. Brancher le modèle de reconnaissance vocale

**État : non fait.** Le dépôt ne contenait ni le modèle ni son format de sortie ; toutes les entrées
utilisées ici sont simulées (`source: "simulated-fixture"` ou `"manual-input"`).

Point d'entrée : `drone_voice/transcription.py` → `TranscriptionEvent`, consommé par
`Pipeline.handle_event` (`drone_voice/pipeline.py`).

```python
from drone_voice.clock import SystemClock
from drone_voice.contract import Confidence
from drone_voice.dedup import JsonlDedupStore
from drone_voice.pipeline import build_pipeline
from drone_voice.transcription import TranscriptionEvent

pipeline = build_pipeline(SystemClock(), dedup_store=JsonlDedupStore("state/processed_command_ids.jsonl"))

def on_model_output(raw):                      # à écrire selon le format réel du modèle
    event = TranscriptionEvent(
        utterance_id=f"{SESSION_ID}:{raw.segment_id}",  # stable pour une prise de parole, unique sinon
        text=raw.text,                                   # tel quel, sans retouche
        is_final=raw.is_final,                           # seule une finale est interprétée
        source="model-xyz-v1",
        confidence=None,                                 # ou Confidence(raw.score, "model-xyz-v1:<nature du score>")
        spoken_at=raw.end_of_speech_utc,                 # si connu : date le message depuis la parole
    )
    message, result = pipeline.handle_event(event)

# appeler pipeline.dispatcher.poll() régulièrement (ex. toutes les 50 ms) pour les fins de durée
```

À vérifier avec le modèle réel :

- Format exact de sortie, encodage et écriture (diacritiques, hamza, chiffres, ponctuation) : adapter
  le vocabulaire en conséquence, sans élargir la correspondance.
- Indicateur de transcription finale. **S'il n'existe pas**, ne pas deviner : appeler
  `Interpreter.interpret_final(text, utterance_id=…)` uniquement quand l'appelant sait que l'énoncé
  est terminé (fin de parole détectée par l'enregistreur, bouton relâché…). Cette limite doit être
  documentée côté application.
- Identifiant de segment/énoncé stable entre les mises à jour, et son comportement au redémarrage.
  Il est aussi la clé de déduplication : deux énoncés distincts ne doivent jamais le partager.
- Heure de fin de parole (`spoken_at`, UTC). Sans elle, le message est daté de sa première
  interprétation : un retard entre la parole et l'interprétation n'est alors pas mesuré.
- Contraintes : `utterance_id` ≤ 256 caractères, `source` ≤ 64 (refus à la création de
  l'événement, avant toute action).
- Un `Dispatcher` construit à la main doit recevoir le vocabulaire (`vocabulary=…`), sinon il refuse
  tout message reconnu (`UNVERIFIED_INTENT`). `build_pipeline` le fait déjà.
- Nature d'un éventuel score de confiance (log-probabilité, score propriétaire…) : il est seulement
  conservé ; tout seuil devra être calibré sur des données réelles.

## 2. Brancher un futur adaptateur de drone

**État : non fait, volontairement.** Seul `SimulatedController` existe. Un adaptateur réel implémente
le protocole `DroneController` (`drone_voice/controller.py`) et est passé à
`build_pipeline(controller=…)`.

Exigences pour l'adaptateur réel :

- `move(command_id, direction, speed_m_s, duration_s)` : translation dans le **repère du drone**
  (droite/gauche = translation latérale, jamais une rotation ; descendre ≠ atterrir) ; borner
  lui-même la durée ; refuser (`accepted=False`) si le mode de vol ne le permet pas.
- `stop(command_id, cause)` : interrompre le déplacement et tenir la position ; **idempotent** ;
  ne jamais désarmer ni couper les moteurs.
- Signaler tout échec (exception ou `accepted=False`) : le répartiteur l'affiche et ne retente
  jamais un `move`.
- Renvoyer `simulated=False` pour distinguer un appel réel d'un appel simulé.

Informations nécessaires avant d'écrire cet adaptateur (inconnues à ce jour) :

1. Autopilote / plateforme (PX4, ArduPilot, SDK constructeur…) et version.
2. Protocole et liaison (MAVLink sur série/UDP, SDK, ROS 2…), latence et fiabilité de la liaison.
3. Mode de vol requis pour des consignes de vitesse (ex. mode guidé/offboard), fréquence minimale
   de rafraîchissement des consignes et comportement en cas d'arrêt du flux.
4. Conventions de repère (body/NED/FRD) et signes des axes.
5. Limites physiques sûres (vitesses, accélérations, altitudes min/max, géorepérage) validées par
   l'équipe de vol — les valeurs de `config/settings.json` ne le sont pas.
6. Comportements de sécurité (failsafe perte de liaison, batterie faible), procédure d'urgence et
   reprise de contrôle manuelle prioritaire.
7. Décollage/atterrissage : hors périmètre ici ; préciser qui les gère.
8. Plan d'essais progressif : SITL (simulateur logiciel de l'autopilote) → HIL → essais captifs en
   environnement contrôlé, avec pilote de sécurité.

## 3. Niveaux de validation

| Niveau | État |
|---|---|
| Tests unitaires | faits (`python -m pytest`) |
| Intégration locale (chaîne complète, adaptateur simulé, horloge contrôlée) | faite |
| Tests avec le modèle de reconnaissance réel | **non faits** (modèle absent) |
| Simulation physique de vol (SITL, Gazebo…) | **non faite** |
| Essais matériels / vol réel | **non faits** — rien ici ne valide un vol réel |
