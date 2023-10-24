// Javascript implementation of a simple, multilayer perceptron neural network.



class NeuralNetwork {
  constructor(hiddenNeuronNum) {
    this.hiddenNeuronNum = hiddenNeuronNum;

    this.w = [];
    this.b = [];
    this.f = [];

    this.__inputActivation = null;
    this.__wActivations = [];
    this.__fActivations = [];

    this.init();
  }

  init() {
    this.w = this.__randomArrayInRange(this.hiddenNeuronNum, -0.5, 0.5);
    this.b = this.__randomArrayInRange(this.hiddenNeuronNum, -0.5, 0.5);
    this.f = this.__randomArrayInRange(this.hiddenNeuronNum, -0.5, 0.5);

    this.__inputActivation = 0;
    this.__wActivations = new Array(this.hiddenNeuronNum).fill(0);
    this.__fActivations = new Array(this.hiddenNeuronNum).fill(0);
  }

  __randomArrayInRange(len, min, max) {
    return (new Array(len)).fill(null).map(() => Math.random() * (max - min) + min);
  }

  relu(x){
    // return Math.max(0, x);
    return x;
  }

  forward(x, updateActivations=false) {
    if (updateActivations) this.__inputActivation = x;

    let y = 0;
    for (let i = 0; i < this.hiddenNeuronNum; i++) {
      let cur = this.w[i] * x + this.b[i];
      if (updateActivations) this.__wActivations[i] = cur;

      cur = this.relu(cur);
      if (updateActivations) this.__fActivations[i] = cur;

      y += cur * this.f[i];
    }

    return y;
  }

  resetActivations() {
    // this.__inputActivation = 0;
    // this.__wActivations = new Array(this.hiddenNeuronNum).fill(0);
    // this.__fActivations = new Array(this.hiddenNeuronNum).fill(0);
  }

  error(y, y_hat) {
    return y - y_hat;
  }

  backwards(error, lr, clipThreshold=1) {
    this.totalHiddenDelta = 0;

    for (let i = 0; i < this.hiddenNeuronNum; i++) {
      let df = error * this.f[i] * (this.__fActivations[i]) * lr;
      this.totalHiddenDelta += df;
      this.f[i] -= df;
    }

    for (let i = 0; i < this.hiddenNeuronNum; i++) {
      let dw = this.totalHiddenDelta * this.w[i] * (this.__inputActivation) * lr;
      this.w[i] -= dw;
    }
  }
}

/*

# Backpropagate error and store in neurons 
def backward_propagate_error(network, expected): 
  for i in reversed(range(len(network))): 
    layer = network[i] 
    errors = list() 
    if i != len(network)-1: 
      for j in range(len(layer)): 
        error = 0.0 
        for neuron in network[i + 1]: 
          error += (neuron['weights'][j] * neuron['delta']) 
        errors.append(error) 
    else: 
      for j in range(len(layer)): 
        neuron = layer[j] 
        errors.append(expected[j] - neuron['output']) 
    for j in range(len(layer)): 
      neuron = layer[j] 
      neuron['delta'] = errors[j] * transfer_derivative(neuron['output']) 

*/


