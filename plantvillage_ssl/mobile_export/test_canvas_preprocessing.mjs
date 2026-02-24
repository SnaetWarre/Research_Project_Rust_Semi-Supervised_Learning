#!/usr/bin/env node
/**
 * Test browser canvas preprocessing simulation
 * 
 * This script simulates what the browser does with canvas.drawImage()
 * and compares it with the expected Python preprocessing.
 */

import { createCanvas, loadImage } from 'canvas';
import { readFileSync } from 'fs';

// ImageNet normalization constants
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

/**
 * Preprocess image using canvas (simulates browser)
 */
async function preprocessWithCanvas(imagePath) {
    // Load image
    const img = await loadImage(imagePath);
    
    // Create canvas and resize to 128x128
    const canvas = createCanvas(128, 128);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 128, 128);
    
    // Get pixel data (RGBA format)
    const imageData = ctx.getImageData(0, 0, 128, 128);
    const pixels = imageData.data;
    
    const floatData = new Float32Array(3 * 128 * 128);
    
    // Track per-channel stats
    const channelSums = [0, 0, 0];
    const channelMins = [Infinity, Infinity, Infinity];
    const channelMaxs = [-Infinity, -Infinity, -Infinity];
    
    // Convert to CHW format with normalization
    for (let y = 0; y < 128; y++) {
        for (let x = 0; x < 128; x++) {
            const srcIdx = (y * 128 + x) * 4; // RGBA format
            for (let c = 0; c < 3; c++) {
                const dstIdx = c * 128 * 128 + y * 128 + x;
                const pixelValue = pixels[srcIdx + c];
                const value = pixelValue / 255.0;
                const normalized = (value - MEAN[c]) / STD[c];
                floatData[dstIdx] = normalized;
                
                // Track stats
                channelSums[c] += normalized;
                if (normalized < channelMins[c]) channelMins[c] = normalized;
                if (normalized > channelMaxs[c]) channelMaxs[c] = normalized;
            }
        }
    }
    
    // Calculate overall stats
    let overallMin = Infinity, overallMax = -Infinity, overallSum = 0;
    for (let i = 0; i < floatData.length; i++) {
        if (floatData[i] < overallMin) overallMin = floatData[i];
        if (floatData[i] > overallMax) overallMax = floatData[i];
        overallSum += floatData[i];
    }
    
    const pixelCount = 128 * 128;
    return {
        overall: {
            min: overallMin,
            max: overallMax,
            mean: overallSum / floatData.length
        },
        channels: [
            {
                name: 'R',
                min: channelMins[0],
                max: channelMaxs[0],
                mean: channelSums[0] / pixelCount
            },
            {
                name: 'G',
                min: channelMins[1],
                max: channelMaxs[1],
                mean: channelSums[1] / pixelCount
            },
            {
                name: 'B',
                min: channelMins[2],
                max: channelMaxs[2],
                mean: channelSums[2] / pixelCount
            }
        ],
        tensor: floatData
    };
}

/**
 * Compare results with expected values
 */
function compareResults(results, expected) {
    console.log('\\n' + '='.repeat(60));
    console.log('Canvas Preprocessing Results (Browser Simulation)');
    console.log('='.repeat(60));
    
    console.log('\\nOverall:');
    console.log(`  min=${results.overall.min.toFixed(3)}, max=${results.overall.max.toFixed(3)}, mean=${results.overall.mean.toFixed(3)}`);
    
    console.log('\\nPer-channel:');
    results.channels.forEach(ch => {
        console.log(`  ${ch.name} channel: min=${ch.min.toFixed(3)}, max=${ch.max.toFixed(3)}, mean=${ch.mean.toFixed(3)}`);
    });
    
    if (!expected) {
        console.log('\\n⚠️  No expected values provided. Run Python test first to get reference values.');
        return;
    }
    
    console.log('\\n' + '='.repeat(60));
    console.log('Comparison with Python Reference');
    console.log('='.repeat(60));
    
    const tolerance = 0.02; // 2% tolerance
    let allPass = true;
    
    // Compare channels
    results.channels.forEach((ch, i) => {
        const expectedMean = expected.channels[i].mean;
        const diff = Math.abs(ch.mean - expectedMean);
        const pass = diff < tolerance;
        allPass = allPass && pass;
        
        const status = pass ? '✅ PASS' : `❌ FAIL (diff: ${diff.toFixed(3)})`;
        console.log(`  ${ch.name} channel mean: ${ch.mean.toFixed(3)} (expected ${expectedMean.toFixed(3)}) ${status}`);
    });
    
    // Compare overall
    const overallDiff = Math.abs(results.overall.mean - expected.overall.mean);
    const overallPass = overallDiff < tolerance;
    allPass = allPass && overallPass;
    
    const status = overallPass ? '✅ PASS' : `❌ FAIL (diff: ${overallDiff.toFixed(3)})`;
    console.log(`  Overall mean: ${results.overall.mean.toFixed(3)} (expected ${expected.overall.mean.toFixed(3)}) ${status}`);
    
    console.log('\\n' + '='.repeat(60));
    if (allPass) {
        console.log('✅ All checks passed! Canvas preprocessing matches Python.');
    } else {
        console.log('❌ Some checks failed. Canvas preprocessing differs from Python.');
        console.log('   This means the browser will get different predictions than expected.');
    }
    console.log('='.repeat(60));
}

// Main
const imagePath = process.argv[2] || '../data/plantvillage/train/Peach___healthy/Peach___healthy_0000.jpg';

console.log('Testing Canvas Preprocessing');
console.log('Image:', imagePath);

// Expected values from Python test (Peach___healthy_0000.jpg)
const expectedPeach = {
    overall: { mean: 0.341 },
    channels: [
        { name: 'R', mean: 0.208 },
        { name: 'G', mean: 0.297 },
        { name: 'B', mean: 0.517 }
    ]
};

const results = await preprocessWithCanvas(imagePath);
compareResults(results, imagePath.includes('Peach___healthy_0000') ? expectedPeach : null);
