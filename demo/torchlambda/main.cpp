





#define VALIDATE_JSON 

#define BASE64 

#define VALIDATE_FIELD 

#define VALIDATE_SHAPE 

#define NORMALIZE 

#define CAST torch::kFloat32

#define DIVIDE 255

#define RETURN_OUTPUT 





#define RETURN_RESULT_ITEM 

#include <string> /* To use for InvalidJson return if any of shape fields not provided */

#include <algorithm>
#include <iterator>

#include <aws/core/Aws.h>
#include <aws/core/utils/base64/Base64.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/core/utils/memory/stl/AWSString.h>

#include <aws/lambda-runtime/runtime.h>

#include <torch/script.h>
#include <torch/torch.h>

using namespace std;

/*!
 *
 *            UTILITY MACROS FOR OUTPUT & RESULT PROCESSING
 *
 */

#define CREATE_JSON_ARRAY(decoded, data, ptr_name, func, torch_type, cpp_type) \
  const auto *ptr_name = data.data_ptr<torch_type>();                          \
  for (int64_t i = 0; i < data.numel(); ++i)                                   \
    decoded[i] = Aws::Utils::Json::JsonValue{}.func(                         \
        (static_cast<cpp_type>(*(ptr_name + i))));

#define ADD_ITEM(value, func, name, torch_type, cpp_type)                      \
  func(name, static_cast<cpp_type>(value.flatten().item<torch_type>()))

/*!
 *
 *                        REQUEST HANDLER
 *
 */

static aws::lambda_runtime::invocation_response
handler(std::shared_ptr<torch::jit::script::Module> &module,
        const aws::lambda_runtime::invocation_request &request
#ifdef BASE64
        ,
        const Aws::Utils::Base64::Base64 &transformer
#endif
) {
    return aws::lambda_runtime::invocation_response::success(
        Aws::Utils::Json::JsonValue{})
    const Aws::String data_field{ "data" };

    /*!
     *
     *               PARSE AND VALIDATE REQUEST
     *
     */

    const auto json = Aws::Utils::Json::JsonValue{request.payload};
    cout<< "here"<<endl;

#ifdef VALIDATE_JSON
    if (!json.WasParseSuccessful())
      return aws::lambda_runtime::invocation_response::failure(
          "Failed to parse request JSON file.", "InvalidJSON");
#endif

    const auto json_view = json.View();
    //cout<< json_view.GetObject(data_field)<<endl;

#ifdef VALIDATE_FIELD
    if (!json_view.KeyExists(data_field))
      return aws::lambda_runtime::invocation_response::failure(
          "Required field: \"" "data" "\" was not provided.", "InvalidJSON");
#ifdef BASE64
    if (!json_view.GetObject(data_field).IsString())
      return aws::lambda_runtime::invocation_response::failure(
          "Required field: \"" "data" "\" is not string.", "InvalidJSON");
#else
    if (!json_view.GetObject(data_field).IsListType())
      return aws::lambda_runtime::invocation_response::failure(
          "Required field: \"" "data" "\" is not list type.", "InvalidJSON");
#endif
#endif

#if not defined(STATIC) && defined(VALIDATE_SHAPE)
    /* Check whether all necessary fields are passed */

    const Aws::String fields[]{"width", "height"};
    for (const auto &field : fields) {
        if (!json_view.KeyExists(field))
          return aws::lambda_runtime::invocation_response::failure(
              "Required input shape field: '" +
                  std::string{field.c_str(), field.size()} +
                  "' was not provided.",
              "InvalidJSON");

        if (!json_view.GetObject(field).IsIntegerType())
          return aws::lambda_runtime::invocation_response::failure(
              "Required shape field: '" +
                  std::string{field.c_str(), field.size()} +
                  "' is not of integer type.",
              "InvalidJSON");
    }

#endif

    /*!
     *
     *            LOAD DATA, TRANSFORM TO TENSOR, NORMALIZE
     *
     */

#ifdef BASE64
    const auto base64_string = json_view.GetString(data_field);
    auto data = transformer.Decode(base64_string);
    auto *data_pointer = data.GetUnderlyingData();
    const std::size_t data_length = data.GetLength();
#else
    const auto nested_json = json_view.GetArray(data_field);
    Aws::Vector<> data;
    data.reserve(nested_json.GetLength());

    for (size_t i = 0; i < nested_json.GetLength(); ++i) {
        data.push_back(static_cast<>(nested_json[i].As()));
    }

    auto *data_pointer = data.data();
    const std::size_t data_length = nested_json.GetLength();
#endif
    const torch::Tensor tensor =
#ifdef NORMALIZE
        torch::data::transforms::Normalize<>{
          {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}
        }(
#endif
            torch::from_blob(
                data_pointer,
                {
                    /* Explicit cast as PyTorch has long int for some reason */
                    static_cast<long>(data_length),
                },
                torch::kUInt8)
                .reshape({1, 3, json_view.GetInteger("width"), json_view.GetInteger("height")})
#ifdef CAST
                .toType(CAST)
#endif
#ifdef DIVIDE
            / DIVIDE
#endif
#ifdef NORMALIZE
        )
#endif
        ;

    /*!
     *
     *              MAKE INFERENCE AND RETURN JSON RESPONSE
     *
     */

    /* Support for multi-output/multi-input? */

    const auto output = module->forward({tensor})
                      .toTensor()
#ifdef RETURN_OUTPUT
                      .toType(torch::kFloat64)
#endif
        ;

    /* Perform operation to create result */
#if defined(RETURN_RESULT) || defined(RETURN_RESULT_ITEM)
    const auto result = (torch::argmax(output,1)).toType(torch::kInt32);
#endif

    /* If array of outputs to be returned gather values as JSON */
#ifdef RETURN_OUTPUT
    Aws::Utils::Array<Aws::Utils::Json::JsonValue> output_array{ 
        static_cast<std::size_t>(output.numel())
    };
    CREATE_JSON_ARRAY(output_array, output, output_ptr, AsDouble,
                      double, double)
#endif

    /* If array of results to be returned gather values as JSON */
#ifdef RETURN_RESULT
    Aws::Utils::Array<Aws::Utils::Json::JsonValue> result_array{
        static_cast<std::size_t>(result.numel())
    };
    CREATE_JSON_ARRAY(result_array, result, result_ptr, AsInteger,
                      int32_t, int)
#endif

    /* Return JSON with response */
    return aws::lambda_runtime::invocation_response::success(
        Aws::Utils::Json::JsonValue{}
#ifdef RETURN_OUTPUT
            .WithArray("output", output_array)
#elif defined(RETURN_OUTPUT_ITEM)
            .ADD_ITEM(output, WithDouble, "output",
                      double, double)
#endif
#ifdef RETURN_RESULT
            .WithArray("result", result_array)
#elif defined(RETURN_RESULT_ITEM)
            .ADD_ITEM(result, WithInteger, "result",
                      int32_t, int)
#endif
            .View()
            .WriteCompact(),
        "application/json"
    );
}

int main() {
    /*!
     *
     *                        INITIALIZE AWS SDK
     *
     */

    Aws::SDKOptions options;
    Aws::InitAPI(options);
    {
#ifndef GRAD
        torch::NoGradGuard no_grad_guard{};
#endif
#ifndef OPTIMIZE
        torch::jit::setGraphExecutorOptimize(false);
#endif

        /* Change name/path to your model if you so desire */
        /* Layers are unpacked to /opt, so you are better off keeping it */
        constexpr auto model_path = "/opt/model.ptc";

        /* You could add some checks whether the module is loaded correctly */
        auto module = Aws::MakeShared<torch::jit::script::Module>(
            "TORCHSCRIPT_MODEL", torch::jit::load(model_path, torch::kCPU));
        if (module == nullptr)
            return -1;
#ifndef GRAD
        module->eval();
#endif

        const Aws::Utils::Base64::Base64 transformer{};
        const auto handler_fn = [&module
#ifdef BASE64
                                 ,
                                 &transformer
#endif
        ](const aws::lambda_runtime::invocation_request &request){
            return handler(module, request
#ifdef BASE64
                           ,
                           transformer
#endif
            );
        };
        aws::lambda_runtime::run_handler(handler_fn);
    }

    Aws::ShutdownAPI(options);
    return 0;
}
