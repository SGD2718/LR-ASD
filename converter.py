import argparse
from collections import OrderedDict

import torch
from test_input import test_input
import coremltools as ct
from ASD import ASD

import torch
import torch.nn as nn
import numpy as np


class AudioEncoder(nn.Module):
    def __init__(self, asd_model: ASD):
        super(AudioEncoder, self).__init__()
        self.audioEncoder = asd_model.model.audioEncoder

    def forward(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        x = self.audioEncoder(x)
        return x


class ASDModel(nn.Module):
    def __init__(self, asd_model: ASD):
        super(ASDModel, self).__init__()
        self.asdModel = asd_model.model
        self.lossV = asd_model.lossV
        self.lossAV = asd_model.lossAV

    def forward(self, visualFeature):
        visualEmbed = self.asdModel.forward_visual_frontend(visualFeature)
        #outsAV = self.asdModel.forward_audio_visual_backend(audioEmbed, visualEmbed)
        #scores = self.lossAV.forward(outsAV)
        outsV = self.asdModel.forward_visual_backend(visualEmbed)
        scores = self.lossV.forward(outsV)
        return scores


def convert_audio_encoder(audio_encoder_model):
    """
    Converts the AudioEncoder model using torch.jit.trace.
    """
    print("--- Starting AudioEncoder Conversion ---")
    try:
        audio_encoder_model.eval()

        # Create a dummy input for tracing using the max sequence length.
        # N=25 -> 4*N = 100
        dummy_audio_input = torch.rand(1, 100, 13)
        traced_audio_encoder = torch.jit.trace(audio_encoder_model, dummy_audio_input)

        # Define symbolic input shape for the final CoreML model
        audio_seq_len = ct.RangeDim(4, 100, default=100)
        audio_input_spec = ct.TensorType(shape=(1, audio_seq_len, 13), name="audioFeatures")

        audio_encoder_mlpackage = ct.convert(
            traced_audio_encoder,
            inputs=[audio_input_spec],
            outputs=[ct.TensorType(name="audioEmbedding")],
            compute_units=ct.ComputeUnit.ALL,
        )
        audio_encoder_mlpackage.save("ASDAudioEncoder.mlpackage")
        print("✅ AudioEncoder.mlpackage saved successfully.")
    except Exception as e:
        print(f"❌ Failed to convert AudioEncoder: {e}")

def convert_asd_model(asd_model_instance):
    """ Converts the main ASDModel using torch.jit.trace. """
    print("\n--- Starting ASDModel Conversion ---")
    try:
        asd_model_instance.eval()

        # Create dummy inputs for tracing using the max sequence length (N=25)
        n_frames_default = 25
        dummy_video_input = torch.rand(1, n_frames_default, 112, 112)
        dummy_audio_embedding = torch.rand(1, n_frames_default, 128)

        # Trace the model. The wrapper correctly defines the single-output inference path.
        #traced_asd_model = torch.jit.trace(asd_model_instance, (dummy_audio_embedding, dummy_video_input))
        traced_asd_model = torch.jit.trace(asd_model_instance, dummy_video_input)

        # Define symbolic input shapes for the final CoreML model
        video_input_spec = ct.TensorType(shape=(1, 25, 112, 112), name="videoInput")
        #audio_embedding_spec = ct.TensorType(shape=(1, 25, 128), name="audioEmbedding")

        # The output spec should not contain a shape, as the converter will infer it.
        scores_output_spec = ct.TensorType(name="scores")

        asd_model_mlpackage = ct.convert(
            traced_asd_model,
            #inputs=[audio_embedding_spec, video_input_spec],
            inputs=[video_input_spec],
            outputs=[scores_output_spec],
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
        )
        asd_model_mlpackage.save("ASDVideoModel.mlpackage")
        print("✅ ASDModel.mlpackage saved successfully.")
    except Exception as e:
        print(f"❌ Failed to convert ASDModel: {e}")

def verify_video_model(args):
    """
    Loads the PyTorch and CoreML video-only models and compares their predictions
    on randomly generated data for various sequence lengths.
    """
    # 1. Load PyTorch Model with trained weights
    print("--- Loading PyTorch model ---")
    complete_model_pt = ASD(device='cpu')
    complete_model_pt.loadParameters(args.weights)
    complete_model_pt.eval()

    video_model_pt = ASDModel(complete_model_pt)
    print("PyTorch video-only model loaded successfully.")

    # 2. Load CoreML Model
    print("\n--- Loading CoreML model ---")
    video_model_ml = ct.models.MLModel(args.asd_model_mlpackage)
    print("CoreML model loaded successfully.")

    # 3. Test with different sequence lengths
    test_video_lengths = [25]
    print(f"\n--- Starting verification for sequence lengths: {test_video_lengths} ---")

    all_passed = True
    for n_frames in test_video_lengths:
        print(f"\n--- Verifying for N = {n_frames} frames ---")

        # a. Generate random video input data
        video_input_np = np.random.rand(1, n_frames, 112, 112).astype(np.float32)

        # b. PyTorch Inference
        with torch.no_grad():
            video_input_pt = torch.from_numpy(video_input_np)
            scores_pt = video_model_pt(video_input_pt)
            scores_pt_np = scores_pt.numpy()

        # c. CoreML Inference
        # Assuming the CoreML model's input is named 'videoInput'
        video_model_input = {'videoInput': video_input_np}
        scores_ml_dict = video_model_ml.predict(video_model_input)
        # Assuming the CoreML model's output is named 'scores'
        scores_ml_np = scores_ml_dict['scores']

        # d. Compare the outputs
        try:
            np.testing.assert_allclose(scores_pt_np, scores_ml_np, rtol=1e-4, atol=1e-5)
            print("✅ Video-only model outputs MATCH.")
        except AssertionError as e:
            print(f"❌ Video-only model outputs DO NOT MATCH for N={n_frames}.")
            print(e)
            all_passed = False

    print("\n--- Verification Summary ---")
    if all_passed:
        print("🎉 All checks passed verification!")
    else:
        print("🔥 One or more checks failed verification.")

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch models to CoreML .mlpackage format.")
    parser.add_argument('--weights', type=str, required=True, help='Path to the .model file with saved weights.')
    args = parser.parse_args()

    print(f"Loading weights from: {args.weights}")

    # Instantiate the parent model that holds all the components
    complete_model = ASD(device='cpu')
    complete_model.loadParameters(args.weights, weights_only=True)
    print("Weights loaded successfully into the complete model.")
    complete_model.eval()

    # Now, extract the sub-models with the loaded weights
    audio_encoder_with_weights = AudioEncoder(complete_model)
    asd_model_with_weights = ASDModel(complete_model)

    print("got the models with loaded weights:")

    # Run the conversion functions on the models that now have the loaded weights
    convert_audio_encoder(audio_encoder_with_weights)
    convert_asd_model(asd_model_with_weights)


def main_verify():
    parser = argparse.ArgumentParser(description="Verify CoreML models against their PyTorch counterparts.")
    parser.add_argument('--weights', type=str, required=True, help='Path to the .model file with saved PyTorch weights.')
    parser.add_argument('--audio-encoder-mlpackage', type=str, default='ASDAudioEncoder.mlpackage', help='Path to the converted AudioEncoder CoreML model.')
    parser.add_argument('--asd-model-mlpackage', type=str, default='ASDVideoModel.mlpackage', help='Path to the converted ASDModel CoreML model.')
    args = parser.parse_args()
    verify_video_model(args)


if __name__ == '__main__':
    # NOTE: You must have an 'ASD.py' file containing the 'ASD' class definition
    # for this script to run.
    main()
    #main_verify()

