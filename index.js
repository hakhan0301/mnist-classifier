const tf = require("@tensorflow/tfjs");
const dataReader = require("./data/data-reader");

const model = createModel(0.17);

// test(10000);
train(0, 10).then(() => test(10000));

function test(count) {
	dataReader.initializeData("./parsed-data/test-data.json");
	const data = tf.tidy(() => dataReader.nextSet(count));
	dataReader.clearData();

	const options = {
		batchSize: 10,
		epochs: 1,
		callbacks: {
			onBatchEnd: (index, logs) => {
				console.log(logs);
				printTensors();
			}
		}
	};

	console.log("started evaluating data");
	model.evaluate(data.features, data.labels).print();
	dataReader.disposeData(data);
	console.log("\n");
}

async function train(fileNumber, count) {
	let filePath = "./parsed-data/train-data-" + fileNumber + ".json";

	dataReader.initializeData(filePath);
	const data = tf.tidy(() => dataReader.nextSet(count));
	dataReader.clearData();

	console.log(tf.memory().numTensors);
	console.log("started training");

	const options = {
		batchSize: 10,
		epochs: 1,
		validationSplit: 0.2,
		metrics: ["loss", "val_loss"],
		callbacks: {
			onBatchEnd: (index, logs) => {
				console.log(logs);
				printTensors();
			}
		}
	};

	const h = await model.fit(data.features, data.labels, options);
	console.log(h.history);
	dataReader.disposeData(data);
	console.log("\n");
}

function createModel(learningRate) {
	const model = tf.sequential();

	model.add(
		tf.layers.conv1d({
			inputShape: [28, 28],
			kernelSize: 5,
			filters: 8,
			strides: 1,
			activation: "relu",
			kernelInitializer: "VarianceScaling"
		})
	);
	model.add(
		tf.layers.maxPooling1d({
			poolSize: 4,
			strides: 4
		})
	);
	model.add(
		tf.layers.conv1d({
			kernelSize: 5,
			filters: 16,
			strides: 1,
			activation: "relu",
			kernelInitializer: "VarianceScaling"
		})
	);
	model.add(
		tf.layers.maxPooling1d({
			poolSize: 2,
			strides: 4
		})
	);
	model.add(tf.layers.flatten());
	model.add(
		tf.layers.dense({
			units: 10,
			kernelInitializer: "VarianceScaling",
			activation: "softmax"
		})
	);

	const optimizer = tf.train.sgd(learningRate);

	model.compile({
		optimizer: optimizer,
		loss: "categoricalCrossentropy"
	});

	return model;
}

function printTensors() {
	console.log("tensor count: " + tf.memory().numTensors);
}
