const inputData = require('./raw-data/mnist_handwritten_test.json');

const fileSystem = require("fs");
const outputFile = './parsed-data/test-data.json';

// data is stored in object as arrays for ease with tensors
const outputObject = { features: [], labels: [] };

for (let i = 0; i < 10000; i++) {
  outputObject.features.push(
    inputData[i].image.map(value => normalize(value))
  );
  outputObject.labels.push(
    inputData[i].label = oneHot(inputData[i].label)
  );
}

const outputData = JSON.stringify(outputObject);
fileSystem.writeFile(outputFile, outputData, (err) => {
    if (err) {
        console.error(err);
        return;
    };
    console.log("File has been created");
});

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
