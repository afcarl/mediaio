import subprocess


def extract_audio_track(input_video_file_path, output_audio_file_path):
    subprocess.check_call(["ffmpeg", "-i", input_video_file_path, "-ac", "1", output_audio_file_path, "-y"])


def merge_video_and_audio_track(input_video_file_path, intput_audio_file_path, output_video_file_path):
    subprocess.check_call(["ffmpeg", "-i", input_video_file_path, "-i", intput_audio_file_path, output_video_file_path, "-y"])
