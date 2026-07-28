# Model interactions

The **Interactions** tab of a model's panel lets you set up relationships between the models you have loaded, so that one model's output can depend on another's. Two kinds are available:

- **Avoidance** — a model can be told to stay away from another within a set distance. For example, a membrane model set to avoid a carbon model within 30 nm will suppress its membrane predictions near the carbon film.
- **Competition** — models can *emit* and *absorb* competition. A model that absorbs competition is suppressed to zero wherever an emitting model predicts a higher value for the same voxel, so where two models overlap the more confident one wins.

!!! note "GUI only"
    Model interactions exist only in the GUI. There is no command-line equivalent, so they have no effect when you segment a dataset with [`ais segment`](../cli/reference.md#ais-segment).

## When to use them

Model interactions were part of the original Ais and are described in the eLife paper; they can be useful for exploring how features relate. Our thinking has since narrowed, though. It helps to separate two jobs: turning a tomogram into a faithful segmentation of a feature, and then *using* those segmentations — combining features, applying constraints, selecting particular particles. Ais (and easymode) increasingly focus on the first: turn a tomogram into a representation of a specific feature, and do the rest afterwards.

So for most work we would keep the segmentation step clean — segment each feature on its own — and do any constraining or combining in a post-processing script, where it is easier to see and adjust. Play with the interactions if you find them useful, but you generally don't need them to get good segmentations.
