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
    
    this.w = __randomArrayInRange(this.hiddenNeuronNum, 0.1, 0.25);
    this.b = __randomArrayInRange(this.hiddenNeuronNum, -100, 200);  
    this.f = __randomArrayInRange(this.hiddenNeuronNum, 0.1, 0.25);

    this.inputZ = 0;
    this.hiddenO = new Array(this.hiddenNeuronNum).fill(0);
    this.hiddenZ = new Array(this.hiddenNeuronNum).fill(0); 
  }

  relu(x){
    return Math.max(0, x);
  }

  reluDerivative(x){
    return (x >= 0 ? x : 0);
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

  backwards(error, lr) {
    this.totalHiddenDelta = 0;

    // update f
    const fClipThreshold = 0.25;
    for (let i = 0; i < this.hiddenNeuronNum; i++) {
      let df = error * this.f[i] * (this.hiddenZ[i]) * lr;

      if (df >  fClipThreshold) df =  fClipThreshold;
      if (df < -fClipThreshold) df = -fClipThreshold;

      this.totalHiddenDelta += df;
      this.f[i] += df;
    }

    // update w
    const wClipThreshold = 0.25;
    for (let i = 0; i < this.hiddenNeuronNum; i++) {
      let dw = this.totalHiddenDelta * this.w[i] * (this.inputZ) * lr;

      if (dw >  wClipThreshold) dw =  wClipThreshold;
      if (dw < -wClipThreshold) dw = -wClipThreshold;

      this.w[i] += dw;
    }

    // update b
    const bClipThreshold = 10;
    for (let i = 0; i < this.hiddenNeuronNum; i++) {
      let db = this.totalHiddenDelta * this.b[i] * 1 * lr;

      if (db >  bClipThreshold) db =  bClipThreshold;
      if (db < -bClipThreshold) db = -bClipThreshold;

      this.b[i] += db;
    }
  }

  train(data, epochs=1, lr=0.0001, debug=false) {
    let trainingError = 0;
    for (let i = 0; i < epochs; i++) {
      trainingError = 0;
      for (let j = 0; j < data.length; j++)  {
        const datapoint = data[j];
  
        const y_hat = this.forward(datapoint.x, updateActivations=true);
        const error = this.error(datapoint.y, y_hat);
        
        this.backwards(error, lr);
        this.resetActivations();

        trainingError += Math.abs(error);
        if (debug) console.log("Datapoint " + j + " (x=" + datapoint.x + ")\n " + "y_hat: " + y_hat + ", y: " + datapoint.y + ", error: " + error);
      }
      if (debug) console.log("epoch " + (i+1) + ": " + trainingError);
      render();
    }
    console.log("Training complete.\nFinal Error: " + trainingError + "\n");
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


