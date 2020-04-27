# iclr2020-notes
personal notes from ICLR2020

## Robustness

- Efficient defence against physically-realizable attack is adversarial training by these recipy [Defending Against Physically Realizable Attacks on Image Classification](http://www.openreview.net/pdf?id=H1xscnEKDr)

- Skip connections improves adv attacks on another networks [Skip Connections Matter: On the Transferability of Adversarial Examples Generated with ResNets](http://www.openreview.net/pdf?id=BJlRs34Fvr)

- (attack-robust activation) k-winners-take-all defence [Enhancing Adversarial Defense by k-Winners-Take-All](http://www.openreview.net/pdf?id=Skgvy64tvr)

- (comments from sofa == unreliable) there is something called "network verification" which is a direction to verify(proove) certain properties of a model (e.g. robustness to certain type of perturbations - rotations, noise, adversarial, etc). The [work 1](http://www.openreview.net/pdf?id=SJxSDxrKDr) have some theoretical proofs(=verification) that adversarial training improves adversarial robustness and [work 2](http://www.openreview.net/pdf?id=B1evfa4tPB) is fast and efficient verification approach
