import os
import torch
import torchaudio

# Project-specific imports
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from mel_processing import spectrogram_torch

def get_text(text, hps):
    """
    Standard VITS text preprocessing:
    1. Convert text to sequence of symbol IDs
    2. Intersperse with blank tokens (token ID 0) if configured
    """
    # Convert text to symbol IDs using configured cleaners
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    
    # VITS often uses blank tokens between every character to improve prosody
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    
    # Convert to torch Tensor
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def enhance_audio(audio_path, net_g, hps, speaker_id=0, device="cuda", noise_scale=0.0, bypass_flow=False, center=True):
    """
    Enhance an existing audio file:
    1. Load with torchaudio
    2. Resample if necessary
    3. Peak normalize to prevent distortion and mismatch
    4. Generate spectrogram
    5. Pass through net_g.voice_enhancement with tuning params
    """
    # Load audio
    wav, sr = torchaudio.load(audio_path)
    
    # Resample if sampling rate doesn't match
    if sr != hps.data.sampling_rate:
        print(f"Resampling audio from {sr}Hz to {hps.data.sampling_rate}Hz...")
        resampler = torchaudio.transforms.Resample(sr, hps.data.sampling_rate)
        wav = resampler(wav)
    
    # Ensure mono
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    
    # Peak Normalization: VITS is sensitive to input volume
    # Normalize to -1.0 to 1.0
    wav_max = wav.abs().max()
    if wav_max > 0:
        wav = wav / wav_max
        
    wav = wav.to(device)
    
    # Generate spectrogram
    # Try center=True if the output sounds "phasey" or metallic
    spec = spectrogram_torch(
        wav, 
        hps.data.filter_length, 
        hps.data.sampling_rate, 
        hps.data.hop_length, 
        hps.data.win_length, 
        center=center
    )
    spec_lengths = torch.LongTensor([spec.size(-1)]).to(device)
    
    # Prepare Speaker ID if needed
    sid = None
    if hps.model.n_speakers > 0:
        sid = torch.LongTensor([speaker_id]).to(device)

    # Perform Voice Enhancement
    print(f"Performing enhancement (noise_scale={noise_scale}, bypass_flow={bypass_flow}, center={center})...")
    with torch.no_grad():
        # returns: o_hat, y_mask, (z, m_q, z_hat)
        o_hat, _, _ = net_g.voice_enhancement(
            spec, 
            spec_lengths, 
            sid=sid, 
            noise_scale=noise_scale, 
            bypass_flow=bypass_flow
        )
        
    return o_hat

def resynthesis_audio(text, ref_audio_path, net_g, hps, speaker_id=0, device="cuda", noise_scale=0.667, center=True):
    """
    Re-synthesize speech: extract durations from reference audio 
    and use them to speak the provided text.
    
    Args:
        text: Text to speak
        ref_audio_path: Path to reference audio (for timing/prosody)
        net_g: The SynthesizerTrn model
        hps: Hyperparameters
        speaker_id: Speaker ID for multi-speaker models
        device: Device to run on
        noise_scale: Sampling randomness (0.0 = deterministic)
        center: STFT center parameter
    
    Returns:
        Output waveform tensor
    """
    # Load and preprocess reference audio
    wav, sr = torchaudio.load(ref_audio_path)
    
    if sr != hps.data.sampling_rate:
        print(f"Resampling reference from {sr}Hz to {hps.data.sampling_rate}Hz...")
        resampler = torchaudio.transforms.Resample(sr, hps.data.sampling_rate)
        wav = resampler(wav)
    
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    
    # Peak normalize
    wav_max = wav.abs().max()
    if wav_max > 0:
        wav = wav / wav_max
    
    wav = wav.to(device)
    
    # Generate spectrogram from reference
    spec = spectrogram_torch(
        wav,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=center
    )
    spec_lengths = torch.LongTensor([spec.size(-1)]).to(device)
    
    # Prepare text
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    x = torch.LongTensor(text_norm).unsqueeze(0).to(device)
    x_lengths = torch.LongTensor([len(text_norm)]).to(device)
    
    # Prepare speaker ID
    sid = None
    if hps.model.n_speakers > 0:
        sid = torch.LongTensor([speaker_id]).to(device)
    
    # Perform resynthesis
    print(f"Performing resynthesis (noise_scale={noise_scale})...")
    with torch.no_grad():
        o, attn, _ = net_g.resynthesis(
            x, 
            x_lengths, 
            spec, 
            spec_lengths, 
            sid=sid, 
            noise_scale=noise_scale
        )
    
    return o, attn

def main():
    # --- Configuration ---
    # Path to your config and checkpoint files
    # Change these to point to your actual files
    config_path = "configs/vctk_base.json" 
    checkpoint_path = "G_latest.pth" # Usually G_*.pth
    
    # Enhancement input (optional)
    input_audio_for_enhancement = "input_to_enhance.wav"
    
    # Input text to be synthesized
    text = "Super VITS is a powerful end to end text to speech model."
    
    # Output file paths
    output_wav = "output_inference.wav"
    output_enhanced_wav = "output_enhanced.wav"
    
    # Speaker ID (sid) is required for multi-speaker models (e.g., VCTK)
    speaker_id = 0 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Hyperparameters
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
        
    print(f"Loading config from {config_path}...")
    hps = utils.get_hparams_from_file(config_path)
    # Safety: Ensure n_speakers is numeric to avoid errors with comparisons
    if getattr(hps.model, 'n_speakers', None) is None:
        hps.model.n_speakers = 0

    # 2. Initialize the Generator Model
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        hop_length=hps.data.hop_length,
        **hps.model).to(device)

    # 3. Load Model Weights
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    _ = utils.load_checkpoint(checkpoint_path, net_g, None)
    net_g.eval()

    # --- Mode 1: Regular TTS Inference ---
    print(f"Preparing text: \"{text}\"")
    text_norm = get_text(text, hps)
    x = text_norm.unsqueeze(0).to(device)
    x_lengths = torch.LongTensor([text_norm.size(0)]).to(device)
    
    sid = None
    if hps.model.n_speakers > 1:
        sid = torch.LongTensor([speaker_id]).to(device)

    print("Synthesizing audio...")
    with torch.no_grad():
        audio, _, _, _ = net_g.infer(x, x_lengths, sid=sid)
    torchaudio.save(output_wav, audio[0].cpu(), hps.data.sampling_rate)
    print(f"TTS saved to: {output_wav}")

    # --- Mode 2: Voice Enhancement ---
    if os.path.exists(input_audio_for_enhancement):
        enhanced_audio = enhance_audio(
            input_audio_for_enhancement, 
            net_g, 
            hps, 
            speaker_id, 
            device,
            noise_scale=0.0,    # 0.0 for deterministic (cleaner), 0.667 for standard
            bypass_flow=False,   # True skips prior flow (often better for very noisy audio)
            center=True         # Toggle if audio sounds misaligned/phasey
        )
        torchaudio.save(output_enhanced_wav, enhanced_audio[0].cpu(), hps.data.sampling_rate)
        print(f"Enhanced audio saved to: {output_enhanced_wav}")
    else:
        print(f"Skipping enhancement: {input_audio_for_enhancement} not found.")

    # --- Mode 3: Resynthesis (Prosody Transfer) ---
    # Uses reference audio timing to re-speak the text
    ref_audio_for_resynthesis = "reference_audio.wav"
    output_resynthesized_wav = "output_resynthesized.wav"
    resynthesis_text = text  # Use the same text, or provide different text
    
    if os.path.exists(ref_audio_for_resynthesis):
        resynth_audio, alignment = resynthesis_audio(
            resynthesis_text,
            ref_audio_for_resynthesis,
            net_g,
            hps,
            speaker_id,
            device,
            noise_scale=0.667,  # 0.0 for deterministic, 0.667 for natural variation
            center=True
        )
        torchaudio.save(output_resynthesized_wav, resynth_audio[0].cpu(), hps.data.sampling_rate)
        print(f"Resynthesized audio saved to: {output_resynthesized_wav}")
    else:
        print(f"Skipping resynthesis: {ref_audio_for_resynthesis} not found.")

    # --- Mode 4: ONNX Export Example ---
    # This demonstrates how to export the model to ONNX for production deployment
    export_onnx_path = "vits_model.onnx"
    print(f"\n--- ONNX Export ---")
    export_onnx(net_g, hps, export_onnx_path, device)


def export_onnx(net_g, hps, output_path, device):
    """
    Exports the VITS Generator to ONNX format.
    
    Why ONNX?
    - Portable format for production deployment (C++, generic runtimes)
    - Often faster inference with ONNX Runtime or TensorRT
    - Removes Python dependency for inference
    
    Critical Step:
    We set net_g.is_export = True. This effectively "patches" the model:
    1. Replaces torch.istft (not ONNX compatible) with a custom implementation.
    2. Simplifies probabilistic paths for deterministic export.
    """
    print(f"Exporting model to {output_path}...")
    
    # 1. Enable Export Mode
    # This triggers the custom ISTFT logic in models.py
    net_g.eval()
    net_g.is_export = True
    
    # 2. Prepare Dummy Inputs
    # ONNX export works by "tracing" the execution with dummy data.
    # The sizes here don't lock the model; we define dynamic axes later.
    dummy_text_length = 50
    dummy_x = torch.randint(low=0, high=len(symbols), size=(1, dummy_text_length), dtype=torch.long).to(device)
    dummy_x_lengths = torch.LongTensor([dummy_text_length]).to(device)
    
    # Scales (defaults)
    noise_scale = 0.667
    noise_scale_w = 0.8
    length_scale = 1.0
    
    # Bundle arguments for the specific forward/infer method signature
    # Signature: infer(x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None)
    args = (dummy_x, dummy_x_lengths)
    kwargs = {
        'noise_scale': noise_scale,
        'length_scale': length_scale,
        'noise_scale_w': noise_scale_w,
    }
    
    # Handle Speaker ID
    if hps.model.n_speakers > 1:
        dummy_sid = torch.LongTensor([0]).to(device)
        kwargs['sid'] = dummy_sid
        input_names = ["text", "text_lengths", "scales", "sid"]
    else:
        input_names = ["text", "text_lengths", "scales"]
        
    # 3. Define Dynamic Axes
    # This tells ONNX which dimensions can change size (e.g., text length)
    dynamic_axes = {
        "text": {0: "batch", 1: "text_length"},
        "text_lengths": {0: "batch"},
        "output": {0: "batch", 2: "audio_length"}
    }
    
    if hps.model.n_speakers > 1:
        dynamic_axes["sid"] = {0: "batch"}

    # 4. Wrap for simplistic export interface
    # torch.onnx.export traces a function. Since `infer` takes kwargs, 
    # and export handles args best, we might need a wrapper if we want to expose 
    # scales as inputs. 
    # For simplicity, we'll export a wrapper Model that takes straightforward inputs.
    
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, text, text_lengths, sid=None):
            # We fix scales for the export to simplify the interface, 
            # or you can make them inputs if you need runtime control.
            return self.model.infer(
                text, 
                text_lengths, 
                sid=sid, 
                noise_scale=noise_scale, 
                length_scale=length_scale, 
                noise_scale_w=noise_scale_w
            )[0] # Return only the audio
            
    wrapped_model = OnnxWrapper(net_g)
    
    # Run Export
    try:
        # Args passed to forward must be positional for export
        export_args = (dummy_x, dummy_x_lengths)
        if hps.model.n_speakers > 1:
            export_args += (dummy_sid,)
            
        torch.onnx.export(
            wrapped_model,
            export_args,
            output_path,
            export_params=True,        # Store the trained parameter weights inside the model file
            opset_version=18,          # Older opsets might fail with complex STFT ops
            do_constant_folding=False,  # Optimization,
            dynamo=False, # Use legacy exporter, dynamo is broken atm
            input_names=input_names if hps.model.n_speakers > 1 else input_names[:-1], # Adjust names
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        print("Success! ONNX model exported.")
        
        # Optional: Verify with onnx library
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model check passed.")
        except ImportError:
            print("Install 'onnx' package to verify the exported model.")
            
    except Exception as e:
        print(f"Export failed: {e}")
    finally:
        # 5. Restore Model State
        net_g.is_export = False
        print("Restored model to native PyTorch mode.")

if __name__ == "__main__":
    main()
