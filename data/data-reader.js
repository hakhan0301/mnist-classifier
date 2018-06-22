const tf = require('@tensorflow/tfjs');

let data, currentIndex;

module.exports = {
  initializeData: function (path) {
    data = require(path);
    currentIndex = 0;
  },

  nextSet: function(count) {
    const set = {
      features: [],
      labels: []
    };

    for(let i = 0; i < count; i++) {
      set.features.push(data.features[currentIndex]);
      set.labels.push(data.labels[currentIndex]);
      currentIndex++;
    }

    set.features = tf.tensor2d(set.features);
    set.labels = tf.tensor2d(set.labels);

    return set;
  },

  getData: function() {
    return data;
  }
}
