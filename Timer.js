let startTime, endTime;

module.exports = {
	startTimer: function() {
		startTime = new Date();
	},

	endTimer: function() {
		let timeDiff = (new Date() - startTime) * 0.001;
		console.log(timeDiff.toFixed(3) + " seconds");
	}
};
