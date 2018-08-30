for (let i = 0; i < 15; i++) {
	let ctx = document.getElementById('canv-' + i).getContext('2d');
	let pixels = digits.features[i];

	for (let x = 0; x < 28; x++) {
		for (let y = 0; y < 28; y++) {
			let currentPixel = pixels[x + y * 28] == 0 ? 4095 : 0;
			let a = pixels[x + y * 28] == 0 ? 0 : 1;
			ctx.fillStyle =
				'rgba(' +
				currentPixel +
				',' +
				currentPixel +
				',' +
				currentPixel +
				',' +
				a +
				')';
			ctx.fillRect(x * 4, y * 4, 4, 4);
		}
	}
}
