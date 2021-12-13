# DJGrad
The goal of this project is to create a network protocol capable of sending machine ML model updates between CAVs that are level 2-4 (human-in-the-loop), without the need for cloud communication or cellular networks.

<p align="center">
  <img src="images/teaser.png" width="600" />
</p>

We are interested in understanding the following aspects of our protocol:

1. **Networking:** How  well do  model gradients  spreadin  realistic  driving  environments?
    - See [veins-dsrc](veins-dsrc) folder
<p align="center">
  <img src="images/dowtown_sim.png" width="600" />
</p>

2. **Learning:** Does our distributed learning protocol positively impact learning?
    - See [learning_sim](learning_sim) folder
<p align="center">
  <img src="images/sim_model.png" width="600" />
</p>

3. **Security & Privacy:** How can a malicious vehicle attack other vehicles by exploiting this distributed learning protocol?
    - See [security](security) folder
<p align="center">
  <img src="images/backdoor_trigger.png" width="600" />
</p>
