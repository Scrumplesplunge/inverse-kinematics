class Vector {
  constructor(x, y) { this.x = x; this.y = y; }
  add(v) { return new Vector(this.x + v.x, this.y + v.y); }
  sub(v) { return new Vector(this.x - v.x, this.y - v.y); }
  neg() { return new Vector(-this.x, -this.y); }
  mul(s) { return new Vector(this.x * s, this.y * s); }
  div(s) { return new Vector(this.x / s, this.y / s); }
  dot(v) { return this.x * v.x + this.y * v.y; }
  squareLength() { return this.dot(this); }
  length() { return Math.sqrt(this.squareLength()); }
  rotate(radians) {
    var c = Math.cos(radians), s = Math.sin(radians);
    return new Vector(c * this.x - s * this.y, s * this.x + c * this.y);
  }
  rotate90() { return new Vector(-this.y, this.x); }
}

class Transform {
  constructor(x, y, offset) { this.x = x; this.y = y; this.offset = offset; }
  static identity() {
    return new Transform(new Vector(1, 0), new Vector(0, 1), new Vector(0, 0));
  }
  static translation(offset) { return Transform.identity().translate(offset); }
  static rotation(radians) { return Transform.identity().rotate(radians); }
  translate(offset) {
    return new Transform(this.x, this.y, this.offset.add(offset));
  }
  rotate(radians) {
    return new Transform(this.x.rotate(radians), this.y.rotate(radians),
                         this.offset.rotate(radians));
  }
  scale(factor) {
    return new Transform(this.x.mul(factor), this.y.mul(factor), this.offset);
  }
  getPosition() { return this.offset; }
  getScale() {
    var squareLength = this.x.squareLength();
    var ratio = squareLength / this.y.squareLength();
    if (ratio < 0.99 || 1.01 < ratio)
      throw new Error("Transform is not affine.");
    return Math.sqrt(squareLength);
  }
  applyToDirection(v) { return this.x.mul(v.x).add(this.y.mul(v.y)); }
  applyToPosition(v) { return this.applyToDirection(v).add(this.offset); }
  after(transform) {
    return new Transform(this.applyToDirection(transform.x),
                         this.applyToDirection(transform.y),
                         this.applyToPosition(transform.offset));
  }
}

class Arm {
  constructor(length, child) {
    this.angle = 0;
    this.length = length;
    this.child = child || null;
  }
  localOffset() { return new Vector(this.length, 0); }
  offset() { return this.localOffset().rotate(this.angle); }
  transform() {
    return Transform.identity()
        .translate(this.localOffset())
        .rotate(this.angle);
  }
  draw(context, baseTransform) {
    var transform = baseTransform.after(this.transform());
    var a = baseTransform.getPosition();
    var b = transform.getPosition();
    var r = this.length * transform.getScale();
    context.save();
      // Draw the arm.
      context.beginPath();
        context.moveTo(a.x, a.y);
        context.lineTo(b.x, b.y);
      context.stroke();
      // Draw a faint impression of its locus.
      context.globalAlpha = 0.1;
      context.beginPath();
        context.arc(a.x, a.y, r, 0, 2 * Math.PI);
      context.stroke();
    context.restore();
    // Draw the next part of the arm.
    if (this.child != null)
      this.child.draw(context, transform);
  }
  joints() { return this.child == null ? 1 : 1 + this.child.joints(); }
  headPosition() {
    return this.transform().applyToPosition(
        this.child == null ? new Vector(0, 0) : this.child.headPosition());
  }
}

// Given an arm, compute the jacobian of the control parameters.
function gradients(arm) {
  var transform = Transform.identity();
  var results = [];
  while (arm != null) {
    var gradient = transform.applyToDirection(arm.offset().rotate90());
    results.push(gradient);
    transform = arm.transform().after(transform);
    arm = arm.child;
  }
  return results;
}

// Given the jacobian of an arm as an array of gradients, compute the
// pseudo-inverse matrix that approximates the necessary input parameters to
// generate a certain velocity.
function pseudoInverse(jacobian) {
  // Compute the value of the jacobian multiplied by its transpose. The result
  // is a symmetric matrix, so only half of the values actually need to be
  // computed; the rest can be copied.
  var a = new Vector(0, 0), b = new Vector(0, 0), c = new Vector(0, 0);
  for (var i = 0, n = jacobian.length; i < n; i++) {
    a.x += jacobian[i].x * jacobian[i].x;
    a.y += jacobian[i].y * jacobian[i].x;
    a.z += jacobian[i].z * jacobian[i].x;
    // b.x is the same as a.y.
    b.y += jacobian[i].y * jacobian[i].y;
    b.z += jacobian[i].z * jacobian[i].y;
    // c.x is the same as a.z.
    // c.y is the same as b.z.
    c.z += jacobian[i].z * jacobian[i].z;
  }
  b.x = a.y;
  c.x = a.z;
  c.y = b.z;
  // Multiply the transpose jacobian by the newly computed matrix.
  var result = [];
  return jacobian.map(x => new Vector(a.dot(x), b.dot(x), c.dot(x)));
}

// Given an arm and a target offset, output the instantaneous parameter
// adjustments necessary to move the head of the arm towards the target offset.
function computeParameters(gradients, target) {
  var inverse = pseudoInverse(gradients);
  var offset = target.sub(arm.headPosition());
  return inverse.map(x => offset.dot(x));
}

// Given a set of gradients (ie. the jacobian) and a set of parameters, return
// the velocity of the head of the corresponding arm.
function computeVelocity(gradients, parameters) {
  if (gradients.length != parameters.length)
    throw new Error("Number of gradients does not match number of parameters.");
  var total = new Vector(0, 0);
  for (var i = 0, n = gradients.length; i < n; i++)
    total = total.add(gradients[i].mul(parameters[i]));
  return total;
}

// Normalize an angle to the range [-Math.PI, Math.PI).
var PI = Math.PI;
var TWO_PI = 2 * PI;
function normalizeAngle(angle) {
  return ((angle + PI) % TWO_PI + TWO_PI) % TWO_PI - PI;
}

// Compute deltas which would cause the arm to smooth out.
function computeSmoothingDeltas(arm) {
  if (arm == null) return [];
  if (arm.child == null) return [0];
  var deltas = [normalizeAngle(arm.child.angle)];
  while (arm.child != null) {
    arm = arm.child;
    // Smooth out relative to parent.
    var selfCorrection = normalizeAngle(-arm.angle);
    if (arm.child == null) {
      deltas.push(selfCorrection);
    } else {
      var childCorrection = normalizeAngle(arm.child.angle);
      deltas.push(normalizeAngle(selfCorrection + 0.2 * childCorrection));
    }
  }
  return deltas;
}

// Apply an array of angular deltas to an arm.
function applyControls(arm, deltas) {
  for (var i = 0, n = deltas.length; i < n; i++) {
    if (arm == null) throw Error("More deltas than arm segments.");
    arm.angle += deltas[i];
    arm = arm.child;
  }
  if (arm != null) throw Error("More arm segments than deltas.");
}

// Given an arm and a target offset, adjust the velocities of the components of
// the arm so that its head will be at the head position. This will iterate to
// improve precision until the head is within the given max range.
function moveArm(arm, target, acceptableRange, maxIterations) {
  for (var iterations = 0; iterations < maxIterations; iterations++) {
    if (target.sub(arm.headPosition()).length() < acceptableRange) break;
    // Compute the adjustments necessary to move in the right direction.
    var jacobian = gradients(arm);
    var parameters = computeParameters(jacobian, target);
    // Apply an exponential backoff so that bones nearer the head move more
    // freely.
    parameters = parameters.map((x, i) => x * Math.pow(1.1, i));
    // Adjust the rate of movement so that it is slow enough to meet the
    // requirements.
    var headVelocity = computeVelocity(jacobian, parameters);
    var speedFactor = 0.1 / headVelocity.length();
    // Update the arm state and recompute the termination condition parameters.
    applyControls(arm, parameters.map(x => x * speedFactor));
  }
}

function smoothArm(arm) {
  // Compute deltas which will smooth out the snake.
  applyControls(arm, computeSmoothingDeltas(arm).map(x => x * 0.1));
}

var arm = null;
var maxBoneLength = 15;
var minBoneLength = 10;
var numBones = 50;
for (var i = 0; i < numBones; i++) {
  var length = minBoneLength + (maxBoneLength - minBoneLength) * i / numBones;
  arm = new Arm(length, arm);
  arm.angle = 4 * Math.PI / numBones;
}
var offset = arm.headPosition();

function updateArm() {
  var center = new Vector(canvas.width / 2, canvas.height / 2);
  smoothArm(arm);
  moveArm(arm, offset, 3, 1000);
  context.clearRect(0, 0, canvas.width, canvas.height);
  arm.draw(context, Transform.translation(center));
}
setInterval(updateArm, 20);

canvas.addEventListener("mousemove", function(event) {
  var center = new Vector(canvas.width / 2, canvas.height / 2);
  var mouse = new Vector(event.offsetX, event.offsetY);
  offset = mouse.sub(center);
});
