#!/usr/bin/env node
"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var chalk = require('chalk');
var clear = require('clear');
var figlet = require('figlet');
var path = require('path');
var program = require('commander');
var fs = require('fs').promises;
var fsRaw = require('fs');
var os = require('os');
var getPixelsLib = require('get-pixels');
var colorMatrix = require("./color_matrix");
var savePixels = require("save-pixels");
var zeros = require("zeros");
program
    .version('0.0.1')
    .description('CLI tool to color correct a directory of underwater images')
    .requiredOption('-d, --dir <dirname>', 'target directory')
    .requiredOption('-o, --output <outdir>', 'output directory')
    .parse(process.argv);
function getPixels(filePath) {
    return __awaiter(this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, new Promise(function (resolve, reject) {
                        getPixelsLib(filePath, function (err, pixels) {
                            if (err) {
                                reject(err);
                            }
                            resolve(pixels);
                        });
                    })];
                case 1: return [2 /*return*/, _a.sent()];
            }
        });
    });
}
var opts = program.opts();
function test() {
    return __awaiter(this, void 0, void 0, function () {
        var files, ps, _i, files_1, file, filePath, destPath, pixels, flatPixelArr, shape, filter, data, i, resultArr, w, h, result, pairs, i_1, row, col, ctr;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, fs.readdir(opts.dir)];
                case 1:
                    files = _a.sent();
                    ps = [];
                    _i = 0, files_1 = files;
                    _a.label = 2;
                case 2:
                    if (!(_i < files_1.length)) return [3 /*break*/, 5];
                    file = files_1[_i];
                    filePath = path.join(opts.dir, file);
                    destPath = path.join(opts.output, file).replace("jpg", "png");
                    return [4 /*yield*/, getPixels(filePath)];
                case 3:
                    pixels = _a.sent();
                    flatPixelArr = [];
                    shape = pixels.shape.slice();
                    filter = colorMatrix(pixels.data, shape[0], shape[1]);
                    data = pixels.data;
                    for (i = 0; i < data.length; i += 4) {
                        data[i] = Math.min(255, Math.max(0, data[i] * filter[0] + data[i + 1] * filter[1] + data[i + 2] * filter[2] + filter[4] * 255)); // Red
                        data[i + 1] = Math.min(255, Math.max(0, data[i + 1] * filter[6] + filter[9] * 255)); // Green
                        data[i + 2] = Math.min(255, Math.max(0, data[i + 2] * filter[12] + filter[14] * 255)); // Blue
                    }
                    resultArr = [];
                    w = shape[0];
                    h = shape[1];
                    result = zeros([w, h, 4]);
                    pairs = [];
                    for (i_1 = 0; i_1 < data.length; i_1 += 4) {
                        pairs.push([data[i_1], data[i_1 + 1], data[i_1 + 2], data[i_1 + 3]]);
                    }
                    row = 0;
                    col = 0;
                    ctr = 0;
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
                        if (col == h) {
                            break;
                        }
                    }
                    savePixels(result, "png").pipe(fsRaw.createWriteStream(destPath));
                    console.log("Saved " + filePath + " to " + destPath);
                    _a.label = 4;
                case 4:
                    _i++;
                    return [3 /*break*/, 2];
                case 5: return [2 /*return*/];
            }
        });
    });
}
test().then(function () { return console.log("Done 🐟"); });
