const fileSystem = require("fs");

// data is stored in object as arrays for ease with tensors

saveTrainingData();

function saveTestingData() {
  const inputData = require('./raw-data/mnist_handwritten_test.json');
  
  const outputObject = formatData(inputData, 0, 10000);
  const outputFile = './parsed-data/test-data.json';

  saveJSON(outputObject, outputFile); 
}

function saveTrainingData() {
  const inputData = require('./raw-data/mnist_handwritten_train.json');

  for (let index = 0; index < 6; index++) {
    const start = index * 10000;
    const end = start + 10000;
    
    const outputObject = formatData(inputData, start, end);
    const outputFile = './parsed-data/train-data-' + index + ".json";

    saveJSON(outputObject, outputFile);
    console.log("file: " + index + " done")
  }
}

function saveJSON(outputObject, outputFile) {
  const outputData = JSON.stringify(outputObject);
  fileSystem.writeFile(outputFile, outputData, (err) => {
      if (err) {
          console.error(err);
          return;
      };
      console.log("File has been created");
  });
}

function formatData(inputData, start, end) {
  const outputObject = { features: [], labels: [] };

  for (let i = start; i < end; i++) {
    outputObject.features.push(
      formatFeatureArray(
        inputData[i].image.map(value => normalize(value))
      )
    );
  
    outputObject.labels.push(
      oneHot(inputData[i].label)
    );
  }

  return outputObject;
}

// input: 1x784 array
// output: 28x28 array
function formatFeatureArray(arr) {
  var output = [];
  while(arr.length > 0) {
    output.push(arr.splice(0,28));
  }
  return output;
}

// input data: 1
// output data: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
function oneHot(value) {
  return Array(10).fill(0).map((e, i) => (i == value) ? 1 : 0);
}

// input data: 0 - 255
// output data: 0 - 1
function normalize(value) {
  // multiplication is used because it's faster than dividing
  // value * (1 / 255)
  return value * 0.00392156862745098;
}
