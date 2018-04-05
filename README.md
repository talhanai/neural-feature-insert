# neural-feature-insert
**mnist_mlp_insert.py** is a script that shows how to model a feedforward neural network with features inserted at different layers. The code was adapted from [this example](https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py). The code utilizes the MNIST dataset, a feedforward network with 2 layers, and evaluates the network with three features sets. 

1. **Raw** features. The standard 784 dimension vector per image.
2. **Even** feature, which is a binary value indicating if the number is even or not.
3. **Greater than or equal to 5** feature, which is a binary feature indicating if the number is greater than or equal to 5 or not.

The script uses the Keras toolkit with the tensorflow back-end.

### Concept
The idea is to model MNIST digits using both the raw information (pixels), as well as features capturing an abstraction, the concept of even/odd and the concept of relative size (greater than), and then to evaluate where these features should be introduced into a network to optimize modeling of the underlying digit. (Given that this is MNIST, results are already super high > 99%, so take this as a toy example, and not something significant to interpret)

![alt-text](https://github.com/talhanai/neural-feature-insert/blob/master/insert.png)

### References

A version of this concept and optimization paradigm was explored here to model the emotional state of individuals engaged in a conversation, and with features from audio, text, and physiologic modalities. (This was originally implemented in MATLAB):

```
@inproceedings{alhanai2017predicting,
  title={Predicting Latent Narrative Mood Using Audio and Physiologic Data.},
  author={AlHanai, Tuka Waddah and Ghassemi, Mohammad Mahdi},
  booktitle={AAAI},
  pages={948--954},
  year={2017}
}

```
