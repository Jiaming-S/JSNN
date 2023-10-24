// Javascript implementation of a simple, multilayer perceptron neural network.



class NeuralNetwork {
  constructor(hiddenNeuronNum) {
    this.hiddenNeuronNum = hiddenNeuronNum;

    this.w = [];
    this.b = [];
    this.f = [];

    this.inputZ = 0;

    this.hiddenO = [];
    this.hiddenZ = [];

    this.init();
  }

  init() {
    const __randomArrayInRange = (len, min, max) => (new Array(len)).fill(null).map(() => Math.random() * (max - min) + min);
    
    this.w = __randomArrayInRange(this.hiddenNeuronNum, -0.5, 0.5);
    this.b = __randomArrayInRange(this.hiddenNeuronNum, -25, 25);
    this.f = __randomArrayInRange(this.hiddenNeuronNum, -0.5, 0.5);

    this.inputZ = 0;
    this.hiddenO = new Array(this.hiddenNeuronNum).fill(0);
    this.hiddenZ = new Array(this.hiddenNeuronNum).fill(0);
  }

  relu(x){
    // return Math.max(0, x);
    return x;
  }

  forward(x, updateActivations=false) {
    if (updateActivations) this.inputZ = x;

    let y = 0;
    for (let i = 0; i < this.hiddenNeuronNum; i++) {
      let cur = this.w[i] * x + this.b[i];
      if (updateActivations) this.hiddenO[i] = cur;

      cur = this.relu(cur);
      if (updateActivations) this.hiddenZ[i] = cur;

      y += cur * this.f[i];
    }

    return y;
  }

  resetActivations() {
    this.inputZ = 0;

    for (let i = 0; i < this.hiddenNeuronNum; i++) {
      this.hiddenO[i] = 0;
      this.hiddenZ[i] = 0;
    }
  }

  error(y, y_hat) {
    return y - y_hat;
  }

  backwards(error, lr, clipThreshold=1) {
    this.totalHiddenDelta = 0;

    for (let i = 0; i < this.hiddenNeuronNum; i++) {
      let df = error * this.f[i] * (this.hiddenZ[i]) * lr;

      if (df > clipThreshold) df = clipThreshold;
      if (df < -clipThreshold) df = -clipThreshold;

      this.totalHiddenDelta += df;
      this.f[i] -= df;
    }

    for (let i = 0; i < this.hiddenNeuronNum; i++) {
      let dw = this.totalHiddenDelta * this.w[i] * (this.inputZ) * lr;

      if (dw > clipThreshold) dw = clipThreshold;
      if (dw < -clipThreshold) dw = -clipThreshold;

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


