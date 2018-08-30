let startTime, endTime, totalTime;

module.exports = {
	startTimer: function() {
		startTime = new Date();
	},

	endTimer: function() {
		let timeDiff = (new Date() - startTime) * 0.001;
		timeDiff = timeDiff.toFixed(3);
		totalTime += timeDiff;
		console.log(timeDiff + ' seconds');
	},

	getTotalTime: function() {
		return totalTime;
	},

	resetTotalTime: function() {
		totalTime = 0;
	}
};
