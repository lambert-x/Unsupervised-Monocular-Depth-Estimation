Our project's code is based on ClementPinard's SfmLearner-Pytorch 
https://github.com/ClementPinard/SfmLearner-Pytorch which implements the system described in the paper:

Unsupervised Learning of Depth and Ego-Motion from Video

[Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), [Matthew Brown](http://matthewalunbrown.com/research/research.html), [Noah Snavely](http://www.cs.cornell.edu/~snavely/), [David G. Lowe](http://www.cs.ubc.ca/~lowe/home.html)

In CVPR 2017 (**Oral**).

See the [project webpage](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/) for more details. 

To accomplish the goal of our project, we add newly-designed architectures (in "models" and "mono2net" folders), new loss functions (in "loss_functions.py") and new training schemes, etc, directly into the original scripts. We also modified required functional scripts for our use.





