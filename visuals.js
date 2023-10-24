
class Screen {
  constructor() {
    this.width = canvas.width;
    this.height = canvas.height;
    
    this.centeredPoint = {};
    this.translatedX = 0;
    this.translatedY = 0;

    this.init();
  }

  init() {
    this.setCenteredPoint(this.width / 2, 0);
  }

  setCenteredPoint(x, y) {
    this.centeredPoint.x = x;
    this.centeredPoint.y = y;
    this.translatedX = this.width / 2 - this.centeredPoint.x;
    this.translatedY = this.height / 2 - this.centeredPoint.y;
  }

  clear() {
    ctx.clearRect(0, 0, this.width, this.height);
  }

  plotPoint(x, y, color='white') {
    x *= domainScaleFactor;
    x += this.translatedX;

    y *= rangeScaleFactor;
    y += this.translatedY;

    ctx.fillStyle = color;
    ctx.fillRect(x, y, 2, 2);
  }

  drawLine(x1, y1, x2, y2, color="#333") {
    x1 *= domainScaleFactor;
    x1 += this.translatedX;

    x2 *= domainScaleFactor;
    x2 += this.translatedX;

    y1 *= rangeScaleFactor;
    y1 += this.translatedY;

    y2 *= rangeScaleFactor;
    y2 += this.translatedY;

    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  generatePointsAlongCurve(pointsNum, curve, domain, fuzz=10) {
    const stepSize = (domain[1] - domain[0]) / pointsNum;

    const points = [];
    for (let i = 0; i < pointsNum; i++) {
      const x = domain[0] + i * stepSize;
      const y = curve(x) + (Math.random() - 0.5) * fuzz;
      points.push({x, y});
    }

    return points;
  }
}



const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');

const domain = [0, 160]; 
const range = [-45, 45];
const domainScaleFactor = canvas.width / (domain[1] - domain[0]); 
const rangeScaleFactor = canvas.height / (range[1] - range[0]); 

const screen = new Screen();
// const nn = new NeuralNetwork(3);
const nn = new NeuralNetwork(1);

const plottedPoints = screen.generatePointsAlongCurve(80, (x) => 3 * Math.cbrt(x * 100) - 50, domain);

function render () {  
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  screen.drawLine(domain[0], 0, domain[1], 0);

  plottedPoints.forEach(point => screen.plotPoint(point.x, point.y));
  
  let lineResolution = 600;
  for (let i = domain[0]; i < domain[1]; i += (domain[1] - domain[0]) / lineResolution) {
    const y = nn.forward(i, updateActivations=false);
    screen.plotPoint(i, y, 'red');
  }
}


function train(lr=0.001, epochs=1) {
  let data = plottedPoints;
  // data = data.sort((a, b) => 0.5 - Math.random());
  
  for (let i = 0; i < epochs; i++) {
    for (let j = 0; j < data.length; j++)  {
      const datapoint = data[j];

      const y_hat = nn.forward(datapoint.x, updateActivations=true);
      const error = nn.error(datapoint.y, y_hat);
      
      nn.backwards(error, lr);
      nn.resetActivations();
  
      break;
    }
  }

  render();
}



(render)();




