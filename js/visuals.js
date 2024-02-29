
class Screen {
  constructor() {
    this.width = canvas.width;
    this.height = canvas.height;
    
    this.centeredPoint = {};
    this.translatedX = 0;
    this.translatedY = 0;

    this.reflectOverXAxis = true;

    this.init();
  }

  init() {
    this.setCenteredPoint(this.width / 2 - 10, this.height / 2 - 10);
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
    if (this.reflectOverXAxis) y = this.height - y;

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
    if (this.reflectOverXAxis) y1 = this.height - y1;

    y2 *= rangeScaleFactor;
    y2 += this.translatedY;
    if (this.reflectOverXAxis) y2 = this.height - y2;

    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
}

function generatePointsAlongCurve(pointsNum, curve, domain, fuzz=0.1) {
  const stepSize = (domain[1] - domain[0]) / pointsNum;

  const points = [];
  for (let i = 0; i < pointsNum; i++) {
    const x = domain[0] + i * stepSize;
    const y = curve(x) + (Math.random() - 0.5) * fuzz;
    points.push({x, y});
  }

  return points;
}



const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');

const domain = [0, 5]; 
const range = [-1, 1];
let domainScaleFactor = canvas.width / (domain[1] - domain[0]); 
let rangeScaleFactor = canvas.height / (range[1] - range[0]); 

const screen = new Screen();
const nn = new NeuralNetwork();

const data = generatePointsAlongCurve(80, (x) => 0.2 * Math.cbrt(x * 100), domain);

function render () {  
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  screen.drawLine(domain[0], 0, domain[1], 0);
  screen.drawLine(0, range[0], 0, range[1]);

  data.forEach(point => screen.plotPoint(point.x, point.y));
  
  let lineResolution = 600;
  for (let i = domain[0]; i < domain[1]; i += (domain[1] - domain[0]) / lineResolution) {
    const y = nn.forward(nj.array(i), grad=false);
    screen.plotPoint(i, y, 'red');
  }
}
render();











let mouseHeld = false;
document.addEventListener('mousedown', (e) => (mouseHeld = true));
document.addEventListener('mouseup', (e) => (mouseHeld = false));
document.addEventListener('mousemove', (e) => {
  if (mouseHeld) {
    let dx = e.movementX;
    let dy = e.movementY;
    if (screen.reflectOverXAxis) dy *= -1;
    
    const x = screen.centeredPoint.x - dx;
    const y = screen.centeredPoint.y - dy;
    screen.setCenteredPoint(x, y);
    render();
  }
});

document.addEventListener('wheel', (e) => {
  const delta = e.deltaY;
  domainScaleFactor += delta / 100;
  rangeScaleFactor += delta / 100;
  render();
});

