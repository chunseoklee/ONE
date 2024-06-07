
#include <string>
#include <memory>
#include <thread>
#include <vector>
#include <iostream>
#include <fstream>

#include "OMTrainingInterpreter.h"
#include "onert-micro.h"

// helper for file processing
using DataBuffer = std::vector<char>;

void readDataFromFile(const std::string &filename, char *data, size_t data_size,
                      size_t start_position = 0)
{
  std::streampos start = start_position;

  std::ifstream fs(filename, std::ifstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");

  fs.seekg(start);

  if (fs.read(data, data_size).fail())
    throw std::runtime_error("Failed to read data from file \"" + filename + "\".\n");
  fs.close();
}

void readDataFromFile(std::ifstream &fs, const std::string &filename, char *data, size_t data_size,
                      size_t start_position = 0)
{
  std::streampos start = start_position;

  fs.seekg(start);

  if (fs.read(data, data_size).fail())
    throw std::runtime_error("Failed to read data from file \"" + filename + "\".\n");
}

void writeDataToFile(const std::string &filename, const char *data, size_t data_size)
{
  std::ofstream fs(filename, std::ofstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");
  if (fs.write(data, data_size).fail())
  {
    throw std::runtime_error("Failed to write data to file \"" + filename + "\".\n");
  }
}

DataBuffer readFile(const char *path)
{
  std::ifstream file(path, std::ios::binary | std::ios::in);
  if (!file.good())
  {
    std::string errmsg = "Failed to open file";
    throw std::runtime_error(errmsg.c_str());
  }

  file.seekg(0, std::ios::end);
  auto fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  // reserve capacity
  DataBuffer model_data(fileSize);

  // read the data
  file.read(model_data.data(), fileSize);
  if (file.fail())
  {
    std::string errmsg = "Failed to read file";
    throw std::runtime_error(errmsg.c_str());
  }

  return model_data;
}

struct nnfw_session
{
private:

public:
  /**
   * @brief Factory method. It creates and initialize nnfw_session
   *
   * @note  Use factory instead of constructor to get status
   */
  static NNFW_STATUS create(nnfw_session **session);

private:
  nnfw_session();

public:
  ~nnfw_session();
  NNFW_STATUS load_model_from_file(const char *package_file_path);
  NNFW_STATUS prepare();
  NNFW_STATUS run();

  NNFW_STATUS set_input(uint32_t index, NNFW_TYPE type, const void *buffer, size_t length);
  NNFW_STATUS set_output(uint32_t index, NNFW_TYPE type, void *buffer, size_t length);

  NNFW_STATUS input_size(uint32_t *number);
  NNFW_STATUS output_size(uint32_t *number);

  NNFW_STATUS set_input_layout(uint32_t index, NNFW_LAYOUT layout);
  NNFW_STATUS set_output_layout(uint32_t index, NNFW_LAYOUT layout);

  NNFW_STATUS set_input_tensorinfo(uint32_t index, const nnfw_tensorinfo *ti);

  NNFW_STATUS input_tensorinfo(uint32_t index, nnfw_tensorinfo *ti);
  NNFW_STATUS output_tensorinfo(uint32_t index, nnfw_tensorinfo *ti);

  //
  // Internal-only API
  //

  NNFW_STATUS set_config(const char *key, const char *value);
  NNFW_STATUS get_config(const char *key, char *value, size_t value_size);
  NNFW_STATUS load_circle_from_buffer(uint8_t *buffer, size_t size);
  NNFW_STATUS load_model_from_modelfile(const char *file_path);

  //
  // Experimental API
  //
  NNFW_STATUS push_pipeline_input(std::vector<void *> *inputs, std::vector<uint32_t> *lengths);
  NNFW_STATUS pop_pipeline_output(std::vector<void *> *outputs);

  NNFW_STATUS input_tensorindex(const char *tensorname, uint32_t *index);
  NNFW_STATUS output_tensorindex(const char *tensorname, uint32_t *index);
  /**
   * @brief   Set backends with string-encoded mapping from operation index to backend type
   *          (cpu, acl_cl)
   */
  NNFW_STATUS set_backends_per_operation(const char *backend_settings);

  NNFW_STATUS train_get_traininfo(nnfw_train_info *info);
  NNFW_STATUS train_set_traininfo(const nnfw_train_info *info);
  NNFW_STATUS train_prepare();
  NNFW_STATUS train_input_tensorinfo(uint32_t index, nnfw_tensorinfo *ti);
  NNFW_STATUS train_expected_tensorinfo(uint32_t index, nnfw_tensorinfo *ti);
  NNFW_STATUS train_set_input(uint32_t index, const void *input,
                              const nnfw_tensorinfo *input_tensorinfo);
  NNFW_STATUS train_set_expected(uint32_t index, const void *expected,
                                 const nnfw_tensorinfo *expected_tensorinfo);
  NNFW_STATUS train_set_output(uint32_t index, NNFW_TYPE type, void *buffer, size_t length);
  NNFW_STATUS train_run(bool update_weights);
  NNFW_STATUS train_get_loss(uint32_t index, float *loss);
  NNFW_STATUS train_export_circle(const char *path);

  NNFW_STATUS train_export_checkpoint(const char *path);
  NNFW_STATUS train_import_checkpoint(const char *path);



private:
  uint32_t getInputSize();
  uint32_t getOutputSize();

private:
  
  onert_micro::OMTrainingInterpreter * _train_interpreter;
  onert_micro::OMConfig _config;
  DataBuffer _model_buf;
  std::string _model_path;
};


nnfw_session::nnfw_session()
  : _train_interpreter{new onert_micro::OMTrainingInterpreter()}
{
  // TODO: Remove after implementing train_set_traininfo
  // Set user defined training settings
  const uint32_t training_epochs = 10;
  const float lambda = 0.01f;
  const uint32_t num_train_layers = 0;
  const onert_micro::OMLoss loss = onert_micro::CROSS_ENTROPY;
  const onert_micro::OMTrainOptimizer train_optim = onert_micro::ADAM;
  const float beta = 0.9;
  const float beta_squares = 0.999;
  const float epsilon = 1e-07;

  _config.train_mode = true;
  {
    onert_micro::OMTrainingContext train_context;
    train_context.batch_size = 1;
    train_context.num_of_train_layers = num_train_layers;
    train_context.lambda = lambda;
    train_context.loss = loss;
    train_context.optimizer = train_optim;
    train_context.beta = beta;
    train_context.beta_squares = beta_squares;
    train_context.epsilon = epsilon;

    _config.training_context = train_context;
  }
}

NNFW_STATUS nnfw_session::create(nnfw_session **session)
{
  if (session == nullptr)
    return NNFW_STATUS_UNEXPECTED_NULL;

  auto new_session = std::unique_ptr<nnfw_session>(new nnfw_session());
  *session = new_session.release();
  

  if (*session == nullptr) {
    return NNFW_STATUS_ERROR;
  }
  
  return NNFW_STATUS_NO_ERROR;
}

nnfw_session::~nnfw_session()
{
  delete _train_interpreter;
}

NNFW_STATUS nnfw_session::load_model_from_file(const char *file_path)
{
  _model_buf = readFile(file_path);
  _config.model_ptr = _model_buf.data();
  _config.model_size = _model_buf.size();
  // TODO: this import should start on nnfw_prepare if inference_interpreter is introduced
  _train_interpreter->importTrainModel(_config.model_ptr, _config);
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::train_prepare()
{
  // TODO: Implement remaining jobs if inference_interpreter is introduced
  // maybe interpreter initialization ? 
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::train_run(bool update_weights)
{
  // TOOD: micro support update_weights ???
  _train_interpreter->trainSingleStep(_config);
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::train_export_circle(const char *path)
{
  _train_interpreter->saveModel(_config, path);
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::train_export_checkpoint(const char *path)
{
  _train_interpreter->saveCheckpoint(_config, path);
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::train_import_checkpoint(const char *path)
{
  _train_interpreter->loadCheckpoint(_config, path);
  return NNFW_STATUS_NO_ERROR;
}


// onert-micr.h implementation

NNFW_STATUS nnfw_create_session(nnfw_session **session)
{
  return nnfw_session::create(session);
}

NNFW_STATUS nnfw_load_model_from_file(nnfw_session *session, const char *package_file_path)
{
  return session->load_model_from_file(package_file_path);
}

NNFW_STATUS nnfw_train_prepare(nnfw_session *session)
{
  return session->train_prepare();
}

NNFW_STATUS nnfw_train(nnfw_session *session, bool update_weights)
{
  return session->train_run(update_weights);
}

NNFW_STATUS nnfw_train_export_circle(nnfw_session *session, const char *path)
{
  return session->train_export_circle(path);  
}

NNFW_STATUS nnfw_train_export_checkpoint(nnfw_session *session, const char *path)
{
  return session->train_export_checkpoint(path);  
}

NNFW_STATUS nnfw_train_import_checkpoint(nnfw_session *session, const char *path)
{
  return session->train_import_checkpoint(path);  
}