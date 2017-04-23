


range_map = lambda input_start, input_end, output_start, output_end: \
	lambda input: output_start + ((output_end - output_start) / (input_end - input_start)) * (input - input_start)

normalize_map = range_map(0, 255, 0, 1)
unnormalize_map = lambda x: int(round(range_map(0, 1, 0, 255)(x)))

static_mode = 'L'
static_num_labels = 4
