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
    this.w = this.__randomArrayInRange(this.hiddenNeuronNum, -0.25, 0.25);
    this.b = this.__randomArrayInRange(this.hiddenNeuronNum, -0.5, 0.5);
    this.f = this.__randomArrayInRange(this.hiddenNeuronNum, -0.25, 0.25);

    this.__inputActivation = 0;
    this.__wActivations = new Array(this.hiddenNeuronNum).fill(0);
    this.__fActivations = new Array(this.hiddenNeuronNum).fill(0);
  }

  __randomArrayInRange(len, min, max) {
    return (new Array(len)).fill(null).map(() => Math.random() * (max - min) + min);
  }

  relu(x){
    return Math.max(0, x);
  }

  forward(x, updateActivations=true) {
    if (updateActivations) this.__inputActivation = x;

    let y = 0;
    for (let i = 0; i < this.hiddenNeuronNum; i++) {
      let cur = this.w[i] * x + this.b[i];
      if (updateActivations) this.__wActivations[i] += cur;

      cur = this.relu(cur);
      if (updateActivations) this.__fActivations[i] += cur;

      y += cur * this.f[i];
    }

    return y;
  }

  resetActivations() {
    this.__inputActivation = 0;
    this.__wActivations = new Array(this.hiddenNeuronNum).fill(0);
    this.__fActivations = new Array(this.hiddenNeuronNum).fill(0);
  }

  __calcError(x, y) {
    return (y - this.forward(x)) ** 2;
  }

  MSE(points) {
    let sum = 0;

    for (let i = 0; i < points.length; i++) {
      const x = points[i].x;
      const y = points[i].y;
      sum += this.__calcError(x, y);
    }

    return sum / points.length;
  }

  backwards(x, y, points, lr, clipThreshold=1) {
    let error = this.MSE(points);
    let grad = error * (this.forward(x, updateActivations=false) - y);

    for (let i = 0; i < this.hiddenNeuronNum; i++) {
      let df = lr * grad * this.__fActivations[i];
      let dw = lr * grad * this.__inputActivation;
      let db = lr * grad;

      const gradientL2Norm = Math.sqrt(df * df + dw * dw + db * db);
      if (gradientL2Norm > clipThreshold) {
        df *= clipThreshold / gradientL2Norm;
        dw *= clipThreshold / gradientL2Norm;
        db *= clipThreshold / gradientL2Norm;
      }

      this.f[i] -= df;

      if (this.__fActivations[i] > 0) {
        this.w[i] -= dw
      }

      this.b[i] -= db
    }
  }
}



