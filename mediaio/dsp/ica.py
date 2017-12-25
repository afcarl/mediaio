import argparse
import os

import numpy as np
from sklearn.decomposition import FastICA

from mediaio.audio_io import AudioSignal


class SignalSeparator:

	@staticmethod
	def decompose_signals(mixed_signals):
		mixed_signals_data = np.stack(mixed_signals, axis=-1)

		ica = FastICA(n_components=len(mixed_signals))
		source_signals = ica.fit_transform(mixed_signals_data)

		return [source_signals[:, i] for i in range(source_signals.shape[-1])]


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_mixed_audio_files", type=str, nargs="*", required=True)
	parser.add_argument("--output_dir", type=str, required=True)
	args = parser.parse_args()

	mixed_audio_signals = [AudioSignal.from_wav_file(f) for f in args.input_mixed_audio_files]
	mixed_signals = [audio_signal.get_data(channel_index=0) for audio_signal in mixed_audio_signals]

	source_signals = SignalSeparator.decompose_signals(mixed_signals)
	for i, source_signal in enumerate(source_signals):
		source_audio_signal = AudioSignal(source_signal, mixed_audio_signals[0].get_sample_rate())
		source_audio_signal.set_sample_type(np.int16)

		source_audio_signal.save_to_wav_file(os.path.join(args.output_dir, "d_%d.wav" % i))

if __name__ == "__main__":
	main()
