# neural-feature-insert
**mnist_mlp_insert.py** is a script that shows how to model a feedforward neural network with features inserted at different layers. The code was adapted from [this example](https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py). The code utilizes the MNIST dataset, and evaluates neural networks with three features sets.

1. **Raw** features. The standard 784 dimension vector per image.
2. **Even** features, which is a binary value indicating if the number is even or not.
3. **Greater than or equal to 5** features, which is a binary feature indicating if the number is greater than or equal to 5 or not.

The idea is to model MNIST digits using both the raw information, as well as features capturing an abstraction, the concept of even/odd and the concept of relative size (greater than), and then to evaluate where these features should be introduced into a network to optimize modeling of the underlying number.

A version of this concept and optimization paradigm was explored here:

```
@inproceedings{alhanai2017predicting,
  title={Predicting Latent Narrative Mood Using Audio and Physiologic Data.},
  author={AlHanai, Tuka Waddah and Ghassemi, Mohammad Mahdi},
  booktitle={AAAI},
  pages={948--954},
  year={2017}
}

```
