# Notes on relevant concepts from literature and sources

[Papers] motherfucker, do you [read] it?

~ frei nach J. Winnfield [Pulp Fiction]



Expert Agent: True Expert Data ("high quality")

...

intermediate realms

enforced state coverage (state crawler)

Non-Expert (Model without perfect runs but minimal pretraining, just to generate better walks for offline training => sentinel model)

Expert + n% random actions (basically epsilon greedy um daten zu generieren)

??? Other options

...

Random Selected Action: Random Walks ("low quality")


=> with 3 to 5 data groups => perform offline training and online evaluation on D4RL 
=> possibly if relevant: use model based agent and check model accuracy after training to determine quality of learned environment model.

=> Model selection still necessary