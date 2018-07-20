const tf = require('@tensorflow/tfjs');
const dataReader = require('./data/data-reader');

dataReader.initializeData('./parsed-data/test-data.json');

const model = createModel(0.07);

test();

async function test () {
  dataReader.initializeData('./parsed-data/train-data-0.json');
  const data = tf.tidy(() => dataReader.nextSet(500));
  // console.log(data);

  console.log(tf.memory().numTensors);
  console.log("started training");
  const options = {
    batchSize: 10,
    epochs: 1, 
    validationSplit: .2,
    callbacks: {
      onBatchEnd: (index, logs) => {
        // console.log(logs);
        console.log(tf.memory().numTensors);
      }
    }
  }

  const h = await model.fit(data.features, data.labels, options);
  console.log(h.history.loss[0]);

  // console.log(tf.memory().numTensors);
  // console.log(tf.memory().numBytes);

  // model.evaluate(tempData, tempOutput).print();
  // model.evaluate(data.features, data.label).print();
}


function createModel(learningRate) {
  const model = tf.sequential();

  model.add(tf.layers.conv1d({
    inputShape: [28, 28],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'VarianceScaling'
  }));
  model.add(tf.layers.maxPooling1d({
    poolSize: 4,
    strides: 4
  }));
  model.add(tf.layers.conv1d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'VarianceScaling'
  }));
  model.add(tf.layers.maxPooling1d({
    poolSize: 2,
    strides: 4
  }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({
    units: 10,
    kernelInitializer: 'VarianceScaling',
    activation: 'softmax'
  }));

  const optimizer = tf.train.sgd(learningRate);  

  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
  });

  return model;
}
