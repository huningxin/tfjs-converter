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

import {loadFrozenModel, NamedTensorMap} from '@tensorflow/tfjs-converter';
import * as tfc from '@tensorflow/tfjs-core';

import {IMAGENET_CLASSES} from './imagenet_classes';

const GOOGLE_CLOUD_STORAGE_DIR =
    'https://storage.googleapis.com/tfjs-models/savedmodel/';
const MODEL_FILE_URL = 'mobilenet_v1_1.0_224/optimized_model.pb';
const WEIGHT_MANIFEST_FILE_URL = 'mobilenet_v1_1.0_224/weights_manifest.json';
const INPUT_NODE_NAME = 'input';
const OUTPUT_NODE_NAME = 'MobilenetV1/Predictions/Reshape_1';
const SCALAR_DIVISOR = 225 / 2;

const TFJS_MODEL_URL = 'web_model/tensorflowjs_model.pb';
const WEIGHTS_MANIFEST_URL = 'web_model/weights_manifest.json';

export class MobileNet {
  constructor() {
    this.PREPROCESS_DIVISOR = tfc.scalar(SCALAR_DIVISOR);
  }

  async load() {
    this.model = await loadFrozenModel(TFJS_MODEL_URL, WEIGHTS_MANIFEST_URL);
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }
  /**
   * Infer through MobileNet. This does standard ImageNet pre-processing before
   * inferring through the model. This method returns named activations as well
   * as softmax logits.
   *
   * @param input un-preprocessed input Array.
   * @return The softmax logits.
   */
  async predict(input) {
    let preprocessedInput;
    if (tfc.getBackend() === 'webgl') {
      preprocessedInput = tfc.div(
          tfc.sub(input.asType('float32'), this.PREPROCESS_DIVISOR),
          this.PREPROCESS_DIVISOR);
    } else {
      const values = input.buffer().values;
      let buffer = new Float32Array(values.length);
      for (let i = 0; i < values.length; ++i) {
        buffer[i] = (values[i] - SCALAR_DIVISOR) / SCALAR_DIVISOR;
      }
      preprocessedInput = tfc.Tensor.make(input.shape, {values: buffer}, 'float32');
    }
    const reshapedInput =
        preprocessedInput.reshape([1, ...preprocessedInput.shape]);
    return await this.model.execute(
        {[INPUT_NODE_NAME]:reshapedInput}, OUTPUT_NODE_NAME);
  }

  async getTopKClasses(predictions, topK) {
    const values = await predictions.data();
    predictions.dispose();

    let predictionList = [];
    for (let i = 0; i < values.length; i++) {
      predictionList.push({value: values[i], index: i});
    }
    predictionList = predictionList
                         .sort((a, b) => {
                           return b.value - a.value;
                         })
                         .slice(0, topK);

    return predictionList.map(x => {
      return {label: IMAGENET_CLASSES[x.index], value: x.value};
    });
  }
}
