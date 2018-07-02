/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';

import {MobileNet} from './mobilenet';
import imageURL from './cat.jpg';

const cat = document.getElementById('cat');
const resultElement = document.getElementById('result');
async function run() {
  resultElement.innerText += 'Loading MobileNet...\n';

  const mobileNet = new MobileNet();
  console.time('Loading of model');
  await mobileNet.load();
  console.timeEnd('Loading of model');

  const pixels = tfc.fromPixels(cat);

  console.time('First prediction');
  let result = await mobileNet.predict(pixels);
  const topK = await mobileNet.getTopKClasses(result, 5);
  console.timeEnd('First prediction');

  topK.forEach(x => {
    resultElement.innerText += `${x.value.toFixed(3)}: ${x.label}\n`;
  });

  let elapsed = 0;
  const iterations = 100;
  for (let i = 0; i < iterations; ++i) {
    console.time(`Subsequent ${i} predictions`);
    const start = performance.now();
    result = await mobileNet.predict(pixels);
    await mobileNet.getTopKClasses(result, 5);
    elapsed += performance.now() - start;
    console.timeEnd(`Subsequent ${i} predictions`);
  }

  const averageTime = elapsed/iterations;
  let averageText = `Average elapsed time: ${averageTime.toFixed(3)} ms\n`;
  resultElement.innerText += averageText
  console.log(averageText);

  mobileNet.dispose();
  return averageTime;
}
cat.onload = async () => {
  resultElement.innerText +='Use WebGL backend\n';
  tfc.setBackend('webgl');
  const webglTime = await run();
  resultElement.innerText += '\n';

  resultElement.innerText +='Use WebML backend\n';
  // As WebML POC API only accepts CPU data, so change the
  // backend to CPU.
  tfc.setBackend('webml');
  const webmlTime = await run();
  resultElement.innerText += '\n';
  const speedupText = `Speedup: ${(webglTime/webmlTime).toFixed(3)}`;
  console.log(speedupText);
  resultElement.innerHTML += '<b>' + speedupText + '</b>';
};
cat.src = imageURL;
