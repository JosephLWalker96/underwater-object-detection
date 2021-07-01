#!/usr/bin/env node

const chalk = require('chalk');
const clear = require('clear');
const figlet = require('figlet');
const path = require('path');
const program = require('commander');
const fs = require('fs').promises;
const fsRaw = require('fs');
const os = require('os');
const getPixelsLib = require('get-pixels');
const colorMatrix = require("./color_matrix");
const savePixels = require("save-pixels")
const zeros = require("zeros");
program
	.version('0.0.1')
	.description('CLI tool to color correct a directory of underwater images')
	.requiredOption('-d, --dir <dirname>', 'target directory')
	.requiredOption('-o, --output <outdir>', 'output directory')
    .requiredOption('-b, --blue_val <blue_val>', 'value for blue color shift') //Joe added 5/17/2021
	.parse(process.argv)

// This is a test
async function getPixels(filePath){
	return await new Promise((resolve, reject) => {
		getPixelsLib(filePath, (err, pixels) => {
			if (err) {
				reject(err);
			}
			resolve(pixels)
		})
	})
}

const opts = program.opts()

async function test() {
	const files = await fs.readdir(opts.dir);
	const ps = [];
	for (let file of files) {
		const filePath = path.join(opts.dir, file);
		const destPath = path.join(opts.output, file).replace("jpg", "png");
		const pixels = await getPixels(filePath);
		const flatPixelArr = [];
		const shape = pixels.shape.slice();

		const filter = colorMatrix(pixels.data, shape[0], shape[1],opts.blue_val);
		const data = pixels.data;
		for (var i = 0; i < data.length; i += 4) {
		    data[i] = Math.min(255, Math.max(0, data[i] * filter[0] + data[i + 1] * filter[1] + data[i + 2] * filter[2] + filter[4] * 255)) // Red
		    data[i + 1] = Math.min(255, Math.max(0, data[i + 1] * filter[6] + filter[9] * 255)) // Green
		    data[i + 2] = Math.min(255, Math.max(0, data[i + 2] * filter[12] + filter[14] * 255)) // Blue
		}
		const resultArr = [];
		const w = shape[0];
		const h = shape[1];

		const result = zeros([w, h, 4]);

		const pairs = [];
		for (let i = 0; i < data.length; i+= 4) {
			pairs.push([data[i], data[i+1], data[i+2], data[i+3]]);
		}
		let row = 0;
		let col = 0;
		let ctr =0;
		while (true) {
			result.set(row, col, 0, pairs[ctr][0]);
			result.set(row, col, 1, pairs[ctr][1]);
			result.set(row, col, 2, pairs[ctr][2]);
			result.set(row, col, 3, pairs[ctr][3]);
			row++;
			ctr++;
			if (row == w) {
				row = 0;
				col++;
			}
			if (col == h){
				break;
			}
		}
		savePixels(result, "png").pipe(fsRaw.createWriteStream(destPath));
		console.log(`Saved ${filePath} to ${destPath}`);
	}
}

test().then(() => console.log("Done 🐟"))
