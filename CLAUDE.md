# EVOwithMS — visualisation Gradio du pipeline MS → VO → Résultats

Repo git séparé. App Gradio de démonstration : **Motion Segmentation Model →
Visual Odometry Model (DEVO) → Résultats**, avec une visualisation gradio des
résultats (génération live ou replay de résultats déjà sauvegardés). Voir le
briefing global dans [`../CLAUDE.md`](../CLAUDE.md).

## Environnement

Conda env : `EVOwithMS`.

## Structure actuelle

- `app.py` : point d'entrée Gradio, `gr.Blocks` avec des `gr.Tab` :
  - "Filtrage EVIMO" → `app/tabs/evimo_viewer.py`
  - "EV-IMO Filtered — Events" → `app/tabs/evs_viewer.py`
  - "Motion Segmentation" → placeholder ("*À venir*")
  - "DEVO Odometry" → placeholder ("*À venir*")
- `app/bag_reader.py`, `app/evs_reader.py` : lecture de données (bags ROS, fichiers
  d'events).
- `DEVO/` : copie/référence du fork DEVO pour cette app.
- `datasets/` : `EV-IMO/`, `EV-IMO_filtered/`, `filter0/` — datasets locaux pour la
  démo.

## À développer (suite du briefing)

- Tab "Motion Segmentation" : visualiser la sortie de `MS_Model` (masque/score
  dynamique) sur des séquences events.
- Tab "DEVO Odometry" : visualiser la trajectoire estimée par DEVO, avec/sans
  motion segmentation.
- Format de stockage des résultats (live vs replay) : **à définir** au moment de
  développer ces tabs — pas encore fixé.
