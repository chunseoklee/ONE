
#include <string>
#include <memory>
#include <thread>
#include <vector>

#include "OMTrainingInterpreter.h"
#include "onert-micro.h"

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
  NNFW_STATUS load_model_from_nnpackage(const char *package_file_path);
  NNFW_STATUS prepare();
  NNFW_STATUS prepare_pipeline(const char *map_file_path);
  NNFW_STATUS run();

  NNFW_STATUS run_async();
  NNFW_STATUS await();

  NNFW_STATUS set_input(uint32_t index, NNFW_TYPE type, const void *buffer, size_t length);
  NNFW_STATUS set_output(uint32_t index, NNFW_TYPE type, void *buffer, size_t length);

  NNFW_STATUS input_size(uint32_t *number);
  NNFW_STATUS output_size(uint32_t *number);

  NNFW_STATUS set_input_layout(uint32_t index, NNFW_LAYOUT layout);
  NNFW_STATUS set_output_layout(uint32_t index, NNFW_LAYOUT layout);

  NNFW_STATUS set_input_tensorinfo(uint32_t index, const nnfw_tensorinfo *ti);

  NNFW_STATUS input_tensorinfo(uint32_t index, nnfw_tensorinfo *ti);
  NNFW_STATUS output_tensorinfo(uint32_t index, nnfw_tensorinfo *ti);

  NNFW_STATUS set_available_backends(const char *backends);
  NNFW_STATUS set_op_backend(const char *op, const char *backend);

  NNFW_STATUS set_workspace(const char *dir);

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

  NNFW_STATUS register_custom_operation(const std::string &id, nnfw_custom_eval eval_func);
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



private:
  uint32_t getInputSize();
  uint32_t getOutputSize();

private:
  
  onert_micro::OMTrainingInterpreter train_interpreter;
  std::string _model_path;
};


nnfw_session::nnfw_session()
  : _{nullptr}, _coptions{onert::compiler::CompilerOptions::fromGlobalConfig()},
    _compiler_artifact{nullptr}, _execution{nullptr}, _kernel_registry{nullptr},
    _train_info{nullptr}, _quant_manager{nullptr}, _codegen_manager{nullptr}, _model_path{""}
{
  // DO NOTHING
}

NNFW_STATUS nnfw_session::create(nnfw_session **session)
{
  if (session == nullptr)
    return NNFW_STATUS_UNEXPECTED_NULL;
  try
  {
    auto new_session = std::unique_ptr<nnfw_session>(new nnfw_session());
    new_session->_kernel_registry = std::make_shared<onert::api::CustomKernelRegistry>();
    *session = new_session.release();
  }
  catch (const std::bad_alloc &e)
  {
    std::cerr << "Error during session creation" << std::endl;
    *session = nullptr; // Set nullptr on error to keep the old behavior
    return NNFW_STATUS_OUT_OF_MEMORY;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during session initialization : " << e.what() << std::endl;
    *session = nullptr; // Set nullptr on error to keep the old behavior
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

nnfw_session::~nnfw_session() = default;
