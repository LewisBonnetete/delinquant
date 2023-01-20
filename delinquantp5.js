let width = window.innerWidth
let height = window.innerHeight
let div = 100
let defaultSize = Math.floor(width / div)
let r
let g
let b
let a
let gradientStep = 1
let wasPressed = false

let video
let poseNet
let poses = []
let confidenceMin = 0.5
let time = 0

let ready = false
let bg = 0
let isFullScreen = false
let parts = [
    'leftAnkle',
    'leftEar',
    'leftElbow',
    'leftEye',
    'leftHip',
    'leftKnee',
    'leftShoulder',
    'leftWrist',
    'nose',
    'rightAnkle',
    'rightEar',
    'rightElbow',
    'rightEye',
    'rightHip',
    'rightKnee',
    'rightShoulder',
    'rightWrist',
]

function generateColor() {
    r = getRandomInt(255)
    g = getRandomInt(255)
    b = getRandomInt(255)
    a = 255
    var obj = {
        "r": r,
        "g": g,
        "b": b
      };
      
      var gradient = '';
      for (var key in obj) {
        if (gradient !== '' && obj[key] < obj[gradient]) {
            gradient = key;
        } else if (gradient === '') {
            gradient = key;
        }
      }
    return { r, g, b, a, gradient, gradientSign: 1 }
}

function getRandomInt(max) {
    return Math.floor(Math.random() * max)
}

function modelLoaded() {
    console.log('Model ready');
}

function videoMap(x, y) {
    return { x: map(x, 0, video.width, 0, width), y: map(y, 0, video.height, 0, height) }
}

function getInfo(pose) {
    let center = { x: 0, y: 0 }
    let minX
    let maxX
    let { length } = pose.keypoints
    
    for (let index = 0; index < length; index++) {
        const point = pose.keypoints[index];
        const { x, y } = videoMap(point.position.x, point.position.y)
        if (index === 0) {
            minX = x
            maxX = x
        } else {
            if (x < minX) minX = x
            if (x > maxX) maxX = x
        }
        center.x += x
        center.y += y
    }
    center.x = center.x / length + 1
    center.y = center.y / length + 1

    const { leftShoulder, rightShoulder } = pose
    const size = dist(rightShoulder.x, rightShoulder.y, leftShoulder.x, leftShoulder.y) * 3

    return { center, minX, maxX, size }
}

function euclideanDistance(pose1, pose2) {
    let distance = 0;
    for (let index = 0; index < parts.length; index++) {
        part = parts[index]
        distance += Math.pow(pose1[part].x - pose2[part].x, 2) + 
                    Math.pow(pose1[part].y - pose2[part].y, 2);
    }
    return Math.sqrt(distance);
}

function findLowestEuclideanDistance(pose1) {
    const lowest = 100000000
    const foundIndex = -1

    for (let index = 0; index < poses.length; index++) {
        const pose2 = poses[index];
        const newEuclideanDistance = euclideanDistance(pose1.pose, pose2.pose)
        if (newEuclideanDistance < lowest && !poses[index].pose.assigned) {
            lowest = newEuclideanDistance
            foundIndex = index
        }
    }

    return { poseIndex: foundIndex, poseScore: lowest }
}

function findPose(pose) {
    return findLowestEuclideanDistance(pose)
}

function drawSkeletons(poses) {
    for (let index = 0; index < poses.length; index++) {
        const { skeleton } = poses[index];

        for (let bone = 0; bone < skeleton.length; bone++) {
            const a = videoMap(skeleton[bone][0].position.x, skeleton[bone][0].position.y);
            const b = videoMap(skeleton[bone][1].position.x, skeleton[bone][1].position.y);
            stroke(255)
            line(a.x, a.y, b.x, b.y)
        }
    }
}

function gotPoses(newPoses) {
    poses.forEach(({ pose }) => pose.assigned = false);
    for (let index = 0; index < newPoses.length; index++) {
        if (newPoses[index].pose.score > confidenceMin) {
            newPoses[index].pose.info = getInfo(newPoses[index].pose)
            let { poseIndex, poseScore } = findPose(newPoses[index]);

            console.log('poseScore', poseScore);

            if (poseIndex > -1 && poseScore < 150) {
                let { gradient } = poses[poseIndex].pose.info.color

                if (poses[poseIndex].pose.info.color[gradient] > 255) poses[poseIndex].pose.info.color.gradientSign = -1
                if (poses[poseIndex].pose.info.color[gradient] < 0) poses[poseIndex].pose.info.color.gradientSign = 1

                poses[poseIndex].pose.info.color[gradient] += gradientStep * poses[poseIndex].pose.info.color.gradientSign
                newPoses[index].pose.info.color = poses[poseIndex].pose.info.color
                newPoses[index].pose.updatedAt = time
                newPoses[index].pose.assigned = true

                poses[poseIndex] = newPoses[index]
            } else {
                newPoses[index].pose.info.color = generateColor()
                newPoses[index].pose.detectedAt = time
                newPoses[index].pose.assigned = true
                poses.push(newPoses[index])
            }
        }
    }
}

function drawBg(val) {
    if (val === undefined) {
        let length = 0 
        while (length <= width) {
            generateColor()
            stroke(r, g, b, a)
            for (let x = length; x <= width; x++) {
                line(x, 0, x, height)
            }
            length += Math.floor(width / div)
        }
    } else {
        background(val)
    }
}

function sortPoses() {
    for (let index = 0; index < poses.length; index++) {
        const { pose } = poses[index];
        if (pose.updatedAt < time - 50 || !pose.assigned) {
            poses.splice(index, 1)
        }
    }
}

function touchStarted () {
    var fs = fullscreen();
    if (!fs && !isFullScreen) {
        isFullScreen = true
        fullscreen(true);
        drawBg(bg)
    }
    ready = true;
  }
  
/* full screening will change the size of the canvas */
function windowResized() {
    resizeCanvas(window.innerWidth, window.innerHeight);
    width = window.innerWidth
    height = window.innerHeight
    drawBg(bg)
}

/* prevents the mobile browser from processing some default
* touch events, like swiping left for "back" or scrolling the page.
*/
document.ontouchmove = function(event) {
    event.preventDefault();
};

function setup() {
    createCanvas(width, height)
    video = createCapture(VIDEO)
    video.hide()
    poseNet = ml5.poseNet(video, modelLoaded)
    poseNet.on('pose', gotPoses)
}

function  draw() {
    if (ready) {
        // image(video, 0, 0)
        // console.log('poses.length', poses.length);
        
        for (let index = 0; index < poses.length; index++) {
            const { pose } = poses[index];
            const { minX, maxX , center: { x, y }, size, color } = pose.info
            noStroke()
            fill(color.r, color.g, color.b, color.a)
            rect(x - size / 2, 0, size, height)
        }
        // drawSkeletons(poses)
        if (mouseIsPressed) {
            if (!wasPressed) {
                noStroke()
                generateColor()
                fill(r, g, b, a)
            }
            rect(mouseX - defaultSize / 2, 0, defaultSize, height)
            wasPressed = true
        } else {
            wasPressed = false
        }
        sortPoses()
        time++
    } else {
        drawBg(bg)
        textAlign(CENTER, TOP);
        textSize(32);
        fill(255)
        text('Click to start', width / 2, height / 2);
    }
}