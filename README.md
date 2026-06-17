## Algorithms

### Mirollo–Strogatz — pulse-coupled synchronization
Integrate-and-fire oscillators on a spatial network. Each charges to threshold,
fires, and nudges its neighbours ahead; the concave response makes the whole
population fire in unison — the "fireflies blinking together" effect.

https://github.com/user-attachments/assets/2e07ef33-0ac4-47c3-8ad1-0addf162710f

📄 Papers:
- Mirollo, Renato E., and Steven H. Strogatz. "Synchronization of pulse-coupled biological oscillators." _SIAM Journal on Applied Mathematics_ 50.6 (1990): 1645–1662.
- Pagliari, Roberto, and Anna Scaglione. "Scalable network synchronization with pulse-coupled oscillators." _IEEE Transactions on Mobile Computing_ 10.3 (2010): 392–405.

### Random voting — consensus
Nodes on a random geometric graph each start with a distinct colour. One at a time,
a node copies a random neighbour's colour. Local agreement spreads until the whole
network reaches a single consensus colour (the voter model).

https://github.com/user-attachments/assets/0e058c65-a788-45af-8744-a916c4238187

📄 Papers:
- Aldous, David. "Random walks on finite groups and rapidly mixing Markov chains." _Séminaire de Probabilités XVII 1981/82: Proceedings_. Berlin, Heidelberg: Springer, 2006. 243–297.
- Pagliari, Roberto, et al. "A simple and scalable algorithm for alignment in broadcast networks." _IEEE Journal on Selected Areas in Communications_ 28.7 (2010): 1190–1199.

### Kuramoto — phase synchronization
Coupled phase oscillators on a ring. Each oscillator nudges the others toward its
phase; above a coupling threshold they spontaneously lock into a common rhythm.
Shown as uncoupled vs. coupled side by side.

https://github.com/user-attachments/assets/7576cc8e-41c4-4443-a726-f3904d3e195c

📄 Papers:
- Kuramoto, Yoshiki. "Collective synchronization of pulse-coupled oscillators and excitable units." _Physica D: Nonlinear Phenomena_ 50.1 (1991): 15–30.

## Rendering

```bash
# one scene
uv run python scripts/run_script.py scripts/<algo>.py --quality fourk --output <name>

# all scenes
uv run python scripts/run_all.py --quality fourk
```
