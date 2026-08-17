# short outline for the report


## Introduction
- es war ein heißer tag in rocky beach und die drei ??? halfen onkel titus beim ausladen eines LKW

## Related Works
- What is CQL? => explain basic principle, no details (maybe adjusted loss functions), (CQL paper)
- results stolen from elsewhere (d3rlpy)
- compared datasets / environments (d4rl vs minari)
- features from literature => just reference, no computation
(schweighofer, swazinna)

## Approach
- CQL model training (parameter / config / formel (Qlearning oder SAC))
- hardware (reproducibilty checklist)
- OUR selected features => how to compute? and why?
- plotting (but thats obvious) => how do we compare performance and features? (e.g. correlation, feature importance in random forest, luca & clustering (teari diagramm))
- WHAT we compare (d4rl vs minari, our CQL vs their CQL, correlation / feature importance, different dataset tiers, different d3rlpy algorithms on d4rl)

## Experiments = Results + Inperpretation
- own training => bad performance
- own training results vs d3rlpy paper results
- comparison of features across sources / environments / datasets / tiers
- relationship between features and performance
  - feature importance (us vs them)
  - feature importance (their different algorithms)
  - correlation
  - clustering
  - forest

## Dicussion = Conclusion + Feature Work
- more training required => give us hardware pls
- predicting performance based on features (mention that we took steps but to lil data)
- importance of features across different training durations
- metriken über verschiedene modelle selber trainieren und noch detaillierter analysieren (grafiken haben wir, aber haben wir halt nicht analysiert)

Osdysseus will return...
In Avengers Doomsday