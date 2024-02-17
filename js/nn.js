
class NeuralNetwork {
  constructor() {
    this.syn0 = nj.random([1, 3]).multiply(10).subtract(5);
    this.syn1 = nj.random([3, 1]).multiply(10).subtract(5);
  }

  forward(x, grad=false, y=null, logging=false) {
    let l1 = this._nonlin(nj.dot(x, this.syn0));
    let l2 = this._nonlin(nj.dot(l1, this.syn1)); 

    if (grad && y) {
      y = nj.array(y).T;

      if (logging) {
        console.log("---NEW EPOCH---");
        console.log("x: ", x.inspect());
        console.log("y: ", y.inspect());
        console.log("l1: ", l1.inspect());
        console.log("l2: ", l2.inspect());
      }

      let l2_delta = (y.subtract(l2)).multiply(this._nonlin(l2, true));
      let l1_delta = l2_delta.dot(this.syn1.T).multiply(this._nonlin(l1, true));


      if (logging) {
        console.log("l2_delta: ", l2_delta.inspect());
        console.log("l1_delta: ", l1_delta.inspect());
      }

      if (logging) {
        console.log("syn0: ", this.syn0.inspect());
        console.log("syn1: ", this.syn1.inspect());
      }

      this.syn1 = this.syn1.add(l1.T.dot(l2_delta));
      this.syn0 = this.syn0.add(x.T.dot(l1_delta));

      if (logging) {
        console.log(this.syn0.inspect());
        console.log(this.syn1.inspect());
      }

      return y.subtract(l2).pow(2).mean();
    }

    return l2.flatten().tolist();
  } 

  _nonlin(x, deriv=false, type='sigmoid') {
    if (type === 'sigmoid') {
      if (deriv) { 
        return x.multiply(nj.ones(x.shape).subtract(x));
      }
      else {
        return nj.sigmoid(x);
      }
    }
    if (type === 'tanh') {
      if (deriv) {
        return nj.ones(x.shape).subtract(nj.tanh(x).multiply(nj.tanh(x)));
      }
      else {
        return nj.tanh(x);
      }
    }
    if (type === 'relu') {
      if (deriv) {
        return nj.clip(x, 0, 999).divide(x);
      }
      else {
        return nj.clip(x, 0, 999);
      }
    }
  }
}



/*


X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0,1,1,0]]).T

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in xrange(60000):
  l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
  l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
  l2_delta = (y - l2)*(l2*(1-l2))
  l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
  syn1 += l1.T.dot(l2_delta)
  syn0 += X.T.dot(l1_delta)


*/
