
const NUM_CLASSES = 26;

const webcam = new Webcam(document.getElementById('webcam'));

const controllerDataset = new ControllerDataset(NUM_CLASSES);

let mobilenet;
let model;

async function loadMobilenet() {
  const mobilenet = await tf.loadModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}


ui.setExampleHandler(label => {
  tf.tidy(() => {
    const img = webcam.capture();
    controllerDataset.addExample(mobilenet.predict(img), label);

    ui.drawThumb(img, label);
  });
});


async function train() {
  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!');
  }

  
  model = tf.sequential({
    layers: [
      
      tf.layers.flatten({inputShape: [7, 7, 256]}),
      tf.layers.dense({
        units: ui.getDenseUnits(),
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  const optimizer = tf.train.adam(ui.getLearningRate());
  
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  
  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * ui.getBatchSizeFraction());
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: ui.getEpochs(),
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        ui.trainStatus('Loss: ' + logs.loss.toFixed(5));
        await tf.nextFrame();
      }
    }
  });
}

let isPredicting = false;

async function predict() {
  ui.isPredicting();
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();

      
      const activation = mobilenet.predict(img);

      
      const predictions = model.predict(activation);

      
      return predictions.as1D().argMax();
    });

    const classId = (await predictedClass.data())[0];
    predictedClass.dispose();

    ui.predictClass(classId);
    await tf.nextFrame();
  }
  ui.donePredicting();
}

document.getElementById('train').addEventListener('click', async () => {
  ui.trainStatus('Training...');
  await tf.nextFrame();
  await tf.nextFrame();
  isPredicting = false;
  train();
});
document.getElementById('predict').addEventListener('click', () => {
  isPredicting = true;
  predict();
});

async function init() {
  await webcam.setup();
  mobilenet = await loadMobilenet();

  
  tf.tidy(() => mobilenet.predict(webcam.capture()));

  ui.init();
}

(function () {
    if (!console) {
        console = {};
    }
    var old = console.log;
    var logger = document.getElementById('log');
    console.log = function (message) {
        if (typeof message == 'object') {
            logger.innerHTML += (JSON && JSON.stringify ? JSON.stringify(message) : String(message)) + '<br />';
        } else {
            logger.innerHTML = message;
        }
    }
})();

init();
