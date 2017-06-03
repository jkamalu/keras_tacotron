class CONFIG:

    num_epochs = 10
    batch_size = 32

    embed_size = 256

    learning_rate = 0.001

    num_conv_regions = 16

    # Audio
    audio_sample_rate = 22050
    audio_window_type = "hann"
    audio_fourier_transform_quantity = 2048
    audio_frame_length = 0.050
    audio_frame_shift = 0.0125
    audio_mel_magnitude_exp = 1.2
    audio_mel_banks = 80
    audio_inversion_iterations = 30

    audio_hop_length = int(audio_sample_rate*audio_frame_shift)
    audio_window_length = int(audio_sample_rate*audio_frame_length)
