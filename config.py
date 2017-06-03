class CONFIG:

    batch_size = 32
    embed_size = 256
    num_conv_regions = 16
    loss_weights = 1

    # Audio
    window_type = "hann"
    fourier_transform_quantity = 2048
    frame_length = 0.050*fourier_transform_quantity
    frame_shift = 0.0125*fourier_transform_quantity
