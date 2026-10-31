# Contrat JSON des commandes — version 1

Source de vérité exécutable : `drone_voice/contract.py` (`validate_message`). L'interpréteur valide
chaque message qu'il émet et le répartiteur revalide chaque message qu'il reçoit avec la même
fonction : contourner l'interpréteur ne contourne pas la validation.

## Encodage

- JSON en UTF-8, arabe écrit lisiblement (pas d'échappement `\uXXXX`).
- Lecture stricte : UTF-8 invalide, BOM, clés dupliquées, `NaN`, `Infinity` et nombres débordant
  vers l'infini sont refusés (`MALFORMED_JSON`).

## Message émis par l'interpréteur / reçu par le répartiteur

Tous les champs sont **obligatoires** (valeur `null` lorsqu'elle est permise). **Politique des
champs supplémentaires : refusés** (`INVALID_STRUCTURE`).

| Champ | Type | Règle |
|---|---|---|
| `version` | entier | `1` exactement (un booléen ou `1.0` est refusé ; autre entier → `UNSUPPORTED_VERSION`) |
| `command_id` | chaîne | ASCII `[A-Za-z0-9][A-Za-z0-9._:-]{0,127}` ; voir « Identifiants » |
| `utterance_id` | chaîne | non vide, ≤ 256 caractères |
| `source` | chaîne | non vide, ≤ 64 ; provenance de la transcription (`simulated-fixture`, `manual-input`, …) |
| `created_at` | chaîne | UTC ISO 8601 avec `Z` : `YYYY-MM-DDTHH:MM:SS[.ffffff]Z` (émis en millisecondes) |
| `expires_at` | chaîne | même format, strictement postérieur à `created_at` |
| `recognized_text` | chaîne ou `null` | transcription originale, **non modifiée** ; `null` seulement si `INVALID_INPUT_TYPE` |
| `normalized_text` | chaîne ou `null` | doit être exactement `normalize_text(recognized_text)` (normalisation v1) |
| `intent` | chaîne ou `null` | `MOVE_FORWARD`, `MOVE_BACKWARD`, `MOVE_RIGHT`, `MOVE_LEFT`, `MOVE_UP`, `MOVE_DOWN`, `STOP` |
| `status` | chaîne | `recognized` ou `rejected` |
| `reason` | chaîne ou `null` | motif de rejet de l'interpréteur (ci-dessous) |
| `confidence` | objet ou `null` | `{"value": nombre fini, "provider": chaîne}` exactement |

Cohérence exigée :

- `recognized` ⇒ `intent` connu, `reason: null`, `normalized_text` non vide.
- `rejected` ⇒ `intent: null` et `reason` parmi :

| `reason` | Signification | Textes |
|---|---|---|
| `EMPTY_INPUT` | vide après normalisation (vide, espaces, ponctuation seule…) | `normalized_text == ""` |
| `INVALID_INPUT_TYPE` | l'entrée n'était pas une chaîne (null, nombre, liste, objet…) | les deux textes `null` |
| `UNKNOWN_EXPRESSION` | la phrase normalisée complète n'est pas dans le vocabulaire (négation, phrase composée, ordres multiples, inconnu) | `normalized_text` non vide |

Exemples : `examples/message_recognized.json`, `examples/message_rejected.json`.

### Statuts : reconnaissance ≠ exécution

| Étape | Où c'est visible |
|---|---|
| Reconnaissance | message `status` (`recognized` / `rejected`) |
| Validation à la réception | résultat du répartiteur `outcome: rejected` + `reason` (codes ci-dessous) |
| Transmission à l'adaptateur | résultat `transmitted: true/false` |
| Échec | `outcome: adapter_failed`, `reason: ADAPTER_ERROR` (jamais présenté comme réussite) |
| Résultat simulé | `outcome: adapter_accepted` **et** `simulated: true` |

### Confiance

Si le modèle fournit un score, il est conservé tel quel avec son `provider`. Il n'est **pas**
supposé être une probabilité (aucune contrainte de plage) et **n'intervient dans aucune décision**
de cette version : aucun seuil n'a été fixé faute de données. Absent ⇒ `null`, jamais inventé.

### Identifiants

`command_id = "cmd-" + uuid5(espace fixe, source + U+001F + utterance_id)`.

- Une même prise de parole (retransmission, nouvelle version de la transcription finale) donne le
  même `command_id` → elle n'est exécutée qu'une fois.
- Deux prises de parole distinctes avec le même texte ont des `utterance_id` différents → deux
  commandes légitimes. Le texte n'est jamais utilisé pour dédupliquer.
- La source doit fournir des `utterance_id` uniques, y compris après un redémarrage du modèle
  (préfixer par un identifiant de session si besoin). `interpret_final` sans `utterance_id` en
  génère un nouveau aléatoire à chaque appel.
- Le répartiteur déduplique sur `command_id` **et** sur la prise de parole (`source`,
  `utterance_id`) : une copie dont un relais aurait réécrit le `command_id` reste un doublon.
  Deux commandes distinctes doivent donc toujours avoir des `utterance_id` distincts.

### Dates d'un message

`created_at` est l'heure de fin de parole si la source la fournit (`TranscriptionEvent.spoken_at`),
sinon l'heure de la **première** interprétation de cette prise de parole. Une finale réémise pour le
même énoncé reprend les mêmes `created_at`/`expires_at` : une retransmission ne rajeunit jamais un
ordre. L'interpréteur mémorise ces dates pour les 4 096 derniers énoncés ; au-delà, seule la
déduplication du répartiteur protège.

## Codes de refus du répartiteur (`DispatchResult.reason` quand `outcome = rejected`)

Ordre des contrôles : JSON → contrat → doublon → vérification texte/intention → fenêtre de
validité → date future → expiration → ordre chronologique → enregistrement de la prise de parole →
action.

Un message valide **consomme** sa prise de parole (`command_id` et clé d'énoncé) lorsqu'il est
exécuté, lorsqu'il est de statut `rejected`, **et aussi lorsqu'il est refusé pour une raison de
temps ou d'ordre** (`INVALID_TIMESTAMP` sur la fenêtre, `FUTURE_TIMESTAMP`, `EXPIRED`,
`OUT_OF_ORDER`). Une retransmission de cet énoncé ne peut donc jamais s'exécuter plus tard. Les refus
structurels (`MALFORMED_JSON`, `INVALID_STRUCTURE`, …) et de vérification (`TEXT_INTENT_MISMATCH`,
`UNVERIFIED_INTENT`) ne consomment rien : une copie intacte reçue ensuite reste traitable.

| Code | Condition |
|---|---|
| `MALFORMED_JSON` | octets non décodables ou JSON invalide/non strict |
| `INVALID_STRUCTURE` | pas un objet, champ manquant/inconnu, mauvais type, valeur hors domaine |
| `UNSUPPORTED_VERSION` | `version` entière ≠ 1 |
| `UNKNOWN_INTENT` | `intent` hors de la liste fermée |
| `INCONSISTENT_MESSAGE` | statut/intention/motif/textes incohérents |
| `INVALID_TIMESTAMP` | date mal formée, `expires_at <= created_at`, ou fenêtre > `max_ttl_s` |
| `FUTURE_TIMESTAMP` | `created_at > maintenant + max_future_skew_s` (égalité acceptée) |
| `EXPIRED` | `maintenant >= expires_at` (à l'instant exact d'expiration, le message est expiré) |
| `DUPLICATE` | `command_id` ou prise de parole (`source`, `utterance_id`) déjà consommés |
| `TEXT_INTENT_MISMATCH` | `normalized_text` ne correspond pas à cette intention dans le vocabulaire du répartiteur |
| `UNVERIFIED_INTENT` | répartiteur construit sans vocabulaire : un message `recognized` ne peut pas être vérifié, il est refusé (sauf `unchecked_text=True` explicite) |
| `OUT_OF_ORDER` | mouvement reconnu créé avant la plus récente commande reconnue acceptée, ou pas strictement après le plus récent STOP accepté |
| `STATE_STORE_ERROR` | impossible d'enregistrer l'identifiant : rien n'est exécuté |

## Résultat du répartiteur

```json
{"kind": "command", "outcome": "adapter_accepted", "command_id": "cmd-…", "intent": "MOVE_FORWARD",
 "reason": null, "transmitted": true, "simulated": true, "detail": "…", "at": "2026-01-01T12:00:00.000Z"}
```

`kind` vaut `command` (réponse à un message), `auto_stop` (arrêt en fin de durée) ou `stop_retry`
(nouvelle tentative d'un STOP échoué). `intent` est l'intention **reconnue** du message ; il vaut
`null` pour `auto_stop` et `stop_retry`, qui ne correspondent à aucune parole.
`outcome` ∈ `rejected`, `not_actionable` (message valide de statut `rejected`), `adapter_accepted`,
`adapter_failed`.

## Politiques du répartiteur

- **Liste fermée** : chaque intention de mouvement correspond à `move(direction=…)` via une table
  figée ; `STOP` correspond à `stop()`. Aucun `eval`, aucun nom de méthode issu de l'entrée.
- **Bornes** : vitesse et durée de chaque intention viennent de `config/settings.json`, validées
  (nombres finis, > 0, ≤ maxima du fichier, eux-mêmes ≤ plafonds codés : 1,0 m/s, 5 s). Valeurs
  réservées à la simulation.
- **Fin de durée** : un mouvement se termine quand `maintenant >= début + duration_s`. `poll()`
  (également exécuté au début de chaque `dispatch`) envoie alors `stop(cause=duration_elapsed)`.
  L'intégrateur doit appeler `poll()` régulièrement ; l'adaptateur réel doit aussi borner lui-même
  la durée (double protection).
- **Concurrence** : pas de file. Une nouvelle commande de mouvement remplace le mouvement actif
  (la dernière commande gagne). `STOP` est traité immédiatement, interrompt le mouvement actif et,
  faute de file, n'a aucune commande en attente à annuler. Il n'attend jamais la fin d'un mouvement.
  Les appels sont sérialisés par un verrou : un `STOP` peut seulement attendre la fin d'un appel
  adaptateur déjà en cours.
- **Ordre** : un mouvement plus ancien que la dernière commande acceptée, ou dont `created_at`
  n'est pas strictement postérieur au dernier STOP accepté (égalité comprise), est refusé
  (`OUT_OF_ORDER`) ; un `STOP` est accepté même en retard, car s'arrêter est l'action sûre.
  L'ordre n'est pas conservé après redémarrage ; la durée de validité borne ce risque.
- **Vérification texte/intention** : le répartiteur a besoin du vocabulaire. Construit sans
  vocabulaire, il refuse les messages `recognized` (`UNVERIFIED_INTENT`) ; s'en passer exige
  `unchecked_text=True` explicite (incompatible avec un vocabulaire fourni).
- **Échecs** : une erreur ou un refus de l'adaptateur donne `adapter_failed`. Un `move` n'est
  **jamais** retenté ; comme il a pu démarrer partiellement, un arrêt de sécurité reste programmé à
  sa fin prévue. Un `stop` échoué est retenté à chaque `poll()` tant qu'un mouvement est suivi
  (`stop` est idempotent), avec sa cause d'origine (`stop_command` pour un STOP, résultat
  `kind: stop_retry`).
- **Durée côté adaptateur** : `SimulatedController` (construit avec une horloge) considère lui-même
  le mouvement terminé à `début + duration_s`, même sans `stop` (propriété `state`). Un adaptateur
  réel doit faire de même.
- **Déduplication** : un message valide consomme sa prise de parole (voir « Codes de refus »)
  **avant** l'appel à l'adaptateur. Garantie : au plus un appel `move` par prise de parole et par
  fichier d'état. Avec le stockage en mémoire, la garantie disparaît au redémarrage (seule la durée
  de validité, 2 s par défaut, limite alors une réexécution). Avec `JsonlDedupStore` (CLI `batch`
  et `dispatch`), elle survit au redémarrage tant que le même fichier est utilisé ; un fichier
  corrompu bloque le démarrage. Un plantage entre l'enregistrement et l'appel perd la commande
  plutôt que de la répéter. Aucune garantie « exactement une fois » n'est offerte vis-à-vis d'un
  drone réel.
- **Transcriptions partielles** : jamais transformées en message, donc jamais transmises.
