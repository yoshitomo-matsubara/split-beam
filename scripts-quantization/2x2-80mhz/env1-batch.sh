CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_2x2-80mhz/htov_968-121-968-norm_mse-epoch40.yaml --log log-q/env1_2x2-80mhz/htov_968-121-968-norm_mse-epoch40.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_2x2-80mhz/htov_968-242-968-norm_mse-epoch40_adam.yaml --log log-q/env1_2x2-80mhz/htov_968-242-968-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_2x2-80mhz/htov_968-121-968-norm_mse-epoch40_adam.yaml --log log-q/env1_2x2-80mhz/htov_968-121-968-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_2x2-80mhz/htov_968-60-968-norm_mse-epoch40_adam.yaml --log log-q/env1_2x2-80mhz/htov_968-60-968-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_2x2-80mhz/htov_968-30-968-norm_mse-epoch40_adam.yaml --log log-q/env1_2x2-80mhz/htov_968-30-968-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_2x2-80mhz/htov_968-3872-3872-1936-1936-968-968-norm_mse-epoch40_adam.yaml --log log-q/env1_2x2-80mhz/htov_968-3872-3872-1936-1936-968-968-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_2x2-80mhz/htov_968-3872-3872-3872-2904-2904-2904-1936-1936-1936-968-968-968-norm_mse-epoch40_adam.yaml --log log-q/env1_2x2-80mhz/htov_968-3872-3872-3872-2904-2904-2904-1936-1936-1936-968-968-968-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_2x2-80mhz/htov_968-3872-7748-7748-3872-968-norm_mse-epoch40_adam.yaml --log log-q/env1_2x2-80mhz/htov_968-3872-7748-7748-3872-968-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_2x2-80mhz/htov_968-7748-7748-3872-3872-1936-1936-968-968-norm_mse-epoch40_adam.yaml --log log-q/env1_2x2-80mhz/htov_968-7748-7748-3872-3872-1936-1936-968-968-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'


