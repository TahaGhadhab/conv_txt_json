"""Interpretation of final Arabic (Tunisian dialect) transcriptions into JSON drone commands.

Pipeline: final transcription -> normalization -> intent -> validation -> JSON message
-> dispatcher -> controller adapter. Only a simulated adapter is provided: nothing in
this package talks to real hardware.
"""

__version__ = "0.1.0"
