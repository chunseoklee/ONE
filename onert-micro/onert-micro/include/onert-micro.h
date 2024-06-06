/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef _ONERT_MICRO_H_
#define _ONERT_MICRO_H_

#ifdef __cplusplus
extern "C" {
#endif


/*
 * typical training flow in onert-micro
 *
 * 1. load model or checkpoint
 *   1-1. (optional) configure training options
 * 2. feed training input / output(e.g. label) data (cf. unit of a step)
 * 3. train a step
 * 4. check loss
 *   4-0. save checkpoint for recovery/resume training
 *   4-1. no more traning -> go to 5
 *   4-2. more training -> go to 2
 * 5. save current state to inference model
 * 6. inference with inference model
// sample example
// 0. create context
nnfw_session *session;
nnfw_create_session(&session);
// 1. load model (and checkpoint if continue training)
nnfw_load_model_from_file(session, MODEL_PATH);
// 1-1. (optional, TBD) configure training options
nnfw_load_ckpt_from_file(session, CKPT_PATH);
nnfw_train_prepare(session);
float training_input[BATCH_SIZE*INPUT_SIZE];
float training_label[BATCH_SIZE*OUTPUT_SIZE];
// main training loop
for(int epoch=0; epoch < NUM_EPOCHS; epoch++) {
  for(int step=0; step < NUM_BATCHES ; step++) {
    // prepare this steps's intput/label
    memcpy(training_input, train_input_data + THIS_BATCH_OFFSET, BATCH_SIZE*INPUT_SIZE);
    memcpy(training_output, train_output_data + THIS_BATCH_OFFSET, BATCH_SIZE*OUTPUT_SIZE);
    // 2. feed training input / expected output
    nnfw_train_set_input(session, 0 , training_input, NULL);
    nnfw_train_set_expected(session, 0 , training_input, NULL);
    // 3. train a step
    nnfw_train(session);
  }
  // 4. check loss
  float loss;
  nnfw_train_get_loss(ctx, 0, &loss);
  if(loss > TARGET_LOSS) {
    nnfw_train_save_as_checkpoint(ctx, CKPT_PATH);
  }
  else {
    nnfw_train_export_circle(ctx, CIRCLE_PATH);
  }
}
*/

typedef struct nnfw_session nnfw_session;

/**
 * @brief Result values returned from a call to an API function
 */
typedef enum
{
  /** Successful */
  NNFW_STATUS_NO_ERROR = 0,
  /**
   * An error code for general use.
   * Mostly used when there is no specific value for that certain situation.
   */
  NNFW_STATUS_ERROR = 1,
  /** Unexpected null argument is given. */
  NNFW_STATUS_UNEXPECTED_NULL = 2,
  /** When a function was called but it is not valid for the current session state. */
  NNFW_STATUS_INVALID_STATE = 3,
  /** When it is out of memory */
  NNFW_STATUS_OUT_OF_MEMORY = 4,
  /** When it was given an insufficient output buffer */
  NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE = 5,
  /** When API is deprecated */
  NNFW_STATUS_DEPRECATED_API = 6,
} NNFW_STATUS;

/**
 * @brief Export current training model into circle model 
 * @note  This function should be called on training mode
 *        This function should be called after {@link nnfw_train}
 *
 * @param[in] session The session to export inference model
 * @param[in] path    The path to export inference model
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_export_circle(nnfw_session *session, const char *path);



#ifdef __cplusplus
}
#endif

#endif //_ONERT_MICRO_H_