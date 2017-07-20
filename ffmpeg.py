import subprocess


def downsample(input_audio_file_path, output_audio_file_path, sample_rate):
    subprocess.check_call(
        ["ffmpeg", "-i", input_audio_file_path, "-ar", str(sample_rate), output_audio_file_path, "-y"]
    )
