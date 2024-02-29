
class NeuralNetwork {
  constructor() {
    this.syn0 = nj.random([2, 3]).multiply(10).subtract(5);
    this.syn1 = nj.random([4, 3]).multiply(10).subtract(5);
    this.syn2 = nj.random([4, 1]).multiply(10).subtract(5);
  }

  forward(x, grad=false, y=null) {
    let appendBiases = (arr) => {
      arr = arr.tolist();
      if (grad) arr.forEach(row => row.push(1));
      if (!grad) arr.push(1);
      return nj.array(arr);
    }
    let removeBiases = (arr) => {
      arr = arr.tolist();
      arr.forEach(row => row.pop());
      return nj.array(arr);
    }
    let extractBiasData = (arr) => {
      arr = arr.tolist();
      let biases = [];
      arr.forEach(row => biases.push(row.pop()));
      return nj.array(biases);
    }
    
    let l0 = x;
    l0 = appendBiases(l0);

    let l1 = this._nonlin(nj.dot(l0, this.syn0));
    l1 = appendBiases(l1);

    let l2 = this._nonlin(nj.dot(l1, this.syn1)); 
    l2 = appendBiases(l2);

    let l3 = this._nonlin(nj.dot(l2, this.syn2)); 

    if (grad && y) {
      y = nj.array(y).T;


      // console.log("layer 0:\n", l0.inspect());
      // console.log("layer 1:\n", l1.inspect());
      // console.log("layer 2:\n", l2.inspect());
      // console.log("layer 3:\n", l3.inspect());
      // console.log("y:\n", y.inspect());

      let l3_error = y.subtract(l3);
      let l3_delta = l3_error.multiply(this._nonlin(l3, true));

      /// --- 
      // console.log(l3_delta.inspect());
      // console.log(this.syn2.T.inspect());  
      // console.log(l3_delta.dot(this.syn2.T).inspect());
      /// ---   

      let l2_error = l3_delta.dot(this.syn2.T);
      let l2_delta = l2_error.multiply(this._nonlin(l2, true));

      /// --- 
      // console.log(l2_delta.inspect());
      // console.log("after remove bias:", removeBiases(l2_delta).inspect());
      // console.log(this.syn1.T.inspect()); 
      // console.log(l2_delta.dot(this.syn1.T).inspect()); 
      /// ---   

      l2_delta = removeBiases(l2_delta);

      let l1_error = l2_delta.dot(this.syn1.T);
      let l1_delta = l1_error.multiply(this._nonlin(l1, true));

      l1_delta = removeBiases(l1_delta);

      this.syn2 = this.syn2.add(l2.T.dot(l3_delta));
      this.syn1 = this.syn1.add(l1.T.dot(l2_delta));
      this.syn0 = this.syn0.add(l0.T.dot(l1_delta));

      return y.subtract(l3).pow(2).mean();
    }
    return l3.flatten().tolist();
  } 

  _nonlin(l0, deriv=false, type='sigmoid') {
    if (type === 'sigmoid') {
      if (deriv) { 
        return l0.multiply(nj.ones(l0.shape).subtract(l0));
      }
      else {
        return nj.sigmoid(l0);
      }
    }
    if (type === 'tanh') {
      if (deriv) {
        return nj.ones(l0.shape).subtract(nj.tanh(l0).multiply(nj.tanh(l0)));
      }
      else {
        return nj.tanh(l0);
      }
    }
    if (type === 'relu') {
      if (deriv) {
        return nj.clip(l0, 0, 999).divide(l0);
      }
      else {
        return nj.clip(l0, 0, 999);
      }
    }
  }
}


/*
...

y = nj.array(y).T;

let l3_error = y.subtract(l3);
let l3_delta = l3_error.multiply(this._nonlin(l3, true));

let l2_error = l3_delta.dot(this.syn2.T);
let l2_delta = l2_error.multiply(this._nonlin(l2, true));

let l1_error = l2_delta.dot(this.syn1.T);
let l1_delta = l1_error.multiply(this._nonlin(l1, true));

this.syn2 = this.syn2.add(l2.T.dot(l3_delta));
this.syn1 = this.syn1.add(l1.T.dot(l2_delta));
this.syn0 = this.syn0.add(l0.T.dot(l1_delta));

...


array([[-0.00114,-0.05755, 0.00406,       0],
       [-0.00009,-0.00583, 0.00727,       0],
       [-0.00009,-0.00666,  0.0049,       0],
       [-0.00131,-0.06314, 0.00488,       0],
       [ -0.0001,-0.00693, 0.00467,       0]]) 

array([[ 4.91129, 4.39238,  0.4196, 2.25817],
       [ 3.55176, 4.67286,-0.90093, 0.66638],
       [ 3.45741,-0.81227, 4.56902, 2.32525]])



*/



/*


l0 = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0,1,1,0]]).T

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in l0range(60000):
  l1 = 1/(1+np.el0p(-(np.dot(l0,syn0))))
  l2 = 1/(1+np.el0p(-(np.dot(l1,syn1))))
  l2_delta = (y - l2)*(l2*(1-l2))
  l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
  syn1 += l1.T.dot(l2_delta)
  syn0 += l0.T.dot(l1_delta)


*/
