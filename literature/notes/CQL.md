CQL augments Bellman error objective with Q-value regularizer
=> "straightforward" to implement
=> outperforms not using it

RL is classically an active procedure, but we now want to learn from previously collected data.
=> this does not work easily
=> again, distributional shift between policy that collected the data and learned policy.

often: erroneously optimistic value function estimates.
=> damage

Anyways, CQL is really cool, can be implemented in less than 20 lines of code and is (of course) the best method for RL (judged by the authors)
No but really, apparantly its quite good

Read the preliminary, its quite good.

CQL refers to Q-learning and Actor-Critic methods.

The CQL Framework (chapter 3)
- heavy maths, but cool if you understand it

At this point the diligent reader notes that he wants to ask the ever so knowledgable ChatGPT for guidance in this ongoing struggle against the math notation in the paper which his pigeon brain is yet unable to comprehend.



To summarize, CQL optimizes a well-defined, penalized empirical RL objective, and performs
high-confidence safe policy improvement over the behavior policy. The extent of improvement is
negatively influenced by higher sampling error, which decays as more samples are observed.
~directly from the end of chapter 3 of the paper

the algorithm looks very simple, basically its Q-learning
it is supposed to be much easier than other offline RL approaches

CQL is better, because other approaches often try to mimic the behaviour policy, which is difficult if data stems from different sources.

Irgendwie nutzen die sehr spezielle Atari Datensätzt? Ist das etwas in der Richtung in der wir das prüfen wollten?