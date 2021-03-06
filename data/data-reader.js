const tf = require('@tensorflow/tfjs');

let data, currentIndex;

module.exports = {
	initializeData: function(path) {
		console.log('started initializing data from file: ' + path);

		data = require(path);
		currentIndex = 0;

		console.log('data initialized');
	},

	// error handling isn't implemented, so that means if count gets high enough, currentIndex can give indexoutofbounds
	nextSet: function(count) {
		console.log('getting next set of size: ' + count);

		const set = {
			features: [],
			labels: []
		};

		for (let i = 0; i < count; i++) {
			// set.features.push(tf.tensor([data.features[currentIndex]], [1, 28, 28]));
			// set.labels.push(tf.tensor(data.labels[currentIndex], [1, 10]));
			set.features.push(data.features[currentIndex]);
			set.labels.push(data.labels[currentIndex]);
			currentIndex++;
		}

		set.features = tf.tensor(set.features, [count, 28, 28]);
		set.labels = tf.tensor2d(set.labels, [count, 10]);

		console.log('got next set');
		return set;
	},

	fullSet: function() {
		return this.nextSet(this.size());
	},

	size: function() {
		return data.labels.length;
	},

	getData: function() {
		return data;
	},

	clearData: function() {
		console.log('cleared data in data-reader');
		data = null;
		currentIndex = 0;
	},

	disposeData: function(data) {
		console.log('disposed of data from object');

		tf.dispose(data.features);
		tf.dispose(data.labels);
	}
};
