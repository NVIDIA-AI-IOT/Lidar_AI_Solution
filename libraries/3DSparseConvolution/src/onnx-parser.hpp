#ifndef ONNX_PARSER_HPP
#define ONNX_PARSER_HPP

#include <spconv/engine.hpp>

namespace spconv{

/**
  Create an engine and load the weights from onnx file

  onnx_file: Store the onnx of model structure, please use tool/deploy/export-scn.py to export the
corresponding onnx precision: What precision to use for model inference. For each layer's precision
should be stored in the "precision" attribute of the layer
            - Model inference will ignore the "precision" attribute of each layer what if set to
Float16
**/
std::shared_ptr<Engine> load_engine_from_onnx(
    const std::string& onnx_file,
    Precision precision = Precision::Float16,
    void* stream = nullptr,
    bool mark_all_output = false);


}; // namespace spconv

#endif // ONNX_PARSER_HPP