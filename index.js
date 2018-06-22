const tf = require('@tensorflow/tfjs');
const dataReader = require('./data/data-reader');

dataReader.initializeData('./parsed-data/test-data.json');

let data = dataReader.nextSet(3);
data.labels.print();

data = dataReader.nextSet(4);
data.labels.print();
