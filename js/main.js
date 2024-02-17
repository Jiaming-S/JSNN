

function main(args=null) {
  let epochs = args?.epochs ?? 10;
  let batch_size = args?.bs ?? 2;

  for (let i = 0; i < epochs; i++) {
    data.sort((a, b) => 0.5 - Math.random());

    let x = nj.array();
    let y = nj.array();
    
    data.forEach(point => {
      x = nj.concatenate(x, point.x);
      y = nj.concatenate(y, point.y);
    });

    let curEpochError = 0;
    for (let j = 0; j < x.shape[0]; j += batch_size) {
      let x_batch = x.slice([j, j + batch_size]).reshape(batch_size, 1);
      let y_batch = y.slice([j, j + batch_size]).reshape(batch_size, 1).T;
      let error = nn.forward(x_batch, true, y_batch);
      curEpochError += error;
    }
    console.log("Epoch: ", i+1, " Error: ", curEpochError);
  }

  render();
}

