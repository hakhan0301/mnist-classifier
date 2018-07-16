const tf = require('@tensorflow/tfjs');
const dataReader = require('./data/data-reader');

dataReader.initializeData('./parsed-data/test-data.json');

const model = createModel(0.07);

test();

async function train (setSize, validationPercentage) {

}

async function test () {
  dataReader.initializeData('./parsed-data/test-data.json');
  const data = tf.tidy(() => dataReader.fullSet());
  console.log(data.features[0]);
  model.execute(data.features[0]);
  // console.log(tf.memory().numTensors);
  // console.log(tf.memory().numBytes);

  // model.evaluate(data.features, data.labels).print();
}

function createModel(learningRate) {
  const output = tf.sequential();

  output.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 4,
    strides: 1,
    filters: 8,
    activation: 'sigmoid'
  }));
  output.add(tf.layers.maxPooling2d({
    poolSize: [2, 2]
  }));
  output.add(tf.layers.conv2d({
    kernelSize: 4,
    strides: 1,
    filters: 8,
    activation: 'sigmoid'
  }));
  output.add(tf.layers.maxPooling2d({
    poolSize: [2, 2]
  }));
  output.add(tf.layers.flatten());
  output.add(tf.layers.dense({
    units: 10,
    activation: 'softmax'
  }));

  output.compile({
    optimizer: tf.train.sgd(learningRate),
    loss: tf.losses.softmaxCrossEntropy,
  });

  return output;
}
