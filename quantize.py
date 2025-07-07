import coremltools as ct
import argparse
import os


def quantize_coreml_model(input_model_path, output_model_path):
    """
    Applies INT8 weight quantization to a Core ML model.

    Args:
        input_model_path (str): Path to the existing Float32 .mlpackage file.
        output_model_path (str): Path to save the new INT8 quantized .mlpackage file.
    """
    if not os.path.exists(input_model_path):
        print(f"Error: Input model file not found at '{input_model_path}'")
        return

    print(f"Loading model from: {input_model_path}")
    # 1. Load the Float32 Core ML model
    model = ct.models.MLModel(input_model_path)

    print("Defining INT8 quantization configuration...")
    # 2. Define the quantization configuration.
    # We will quantize the weights of 'conv' and 'linear' layers to 8 bits.
    op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",  # Use symmetric quantization
        dtype="int8",  # Target 8-bit integer
        weight_threshold=512  # Quantize layers with more than 512 weights
    )

    # Create a global config object.
    # By default, this configuration applies to all supported op types (like conv, linear).
    config = ct.optimize.coreml.OptimizationConfig(
        global_config=op_config
    )

    print("Applying INT8 quantization to model weights...")
    # 3. Apply the quantization to the model
    # This function returns a new, quantized model
    quantized_model = ct.optimize.coreml.linear_quantize_weights(
        model,
        config=config
    )

    # 4. Get the spec of the quantized model to update its metadata
    spec = quantized_model.get_spec()

    # Update the description to indicate it's quantized
    if spec.description.metadata.shortDescription:
        new_description = spec.description.metadata.shortDescription + " (INT8 Quantized)"
        ct.utils.rename_feature(spec, spec.description.output[0].name, spec.description.output[0].name + "_INT8")  # Simple way to set description
        quantized_model.short_description = new_description

    # 5. Save the new quantized model
    print(f"Saving quantized model to: {output_model_path}")
    quantized_model.save(output_model_path)
    print("Quantization complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Quantize a Float32 Core ML model (.mlpackage) to INT8 weights.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="Path to the input Float32 .mlpackage model file."
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="Path to save the output INT8 quantized .mlpackage model file."
    )
    args = parser.parse_args()

    quantize_coreml_model(args.input, args.output)