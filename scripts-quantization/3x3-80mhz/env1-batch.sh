CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-80mhz/htov_1452-181-1452-norm_mse-epoch40.yaml --log log-q/env1_3x3-80mhz/htov_1452-181-1452-norm_mse-epoch40.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-80mhz/htov_1452-363-1452-norm_mse-epoch40_adam.yaml --log log-q/env1_3x3-80mhz/htov_1452-363-1452-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-80mhz/htov_1452-181-1452-norm_mse-epoch40_adam.yaml --log log-q/env1_3x3-80mhz/htov_1452-181-1452-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-80mhz/htov_1452-90-1452-norm_mse-epoch40_adam.yaml --log log-q/env1_3x3-80mhz/htov_1452-90-1452-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-80mhz/htov_1452-45-1452-norm_mse-epoch40_adam.yaml --log log-q/env1_3x3-80mhz/htov_1452-45-1452-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-80mhz/htov_1452-5808-5808-2904-2904-1452-1452-norm_mse-epoch40_adam.yaml --log log-q/env1_3x3-80mhz/htov_1452-5808-5808-2904-2904-1452-1452-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-80mhz/htov_1452-5808-5808-5808-2904-2904-2904-2904-2904-2904-1452-1452-1452-norm_mse-epoch40_adam.yaml --log log-q/env1_3x3-80mhz/htov_1452-5808-5808-5808-2904-2904-2904-2904-2904-2904-1452-1452-1452-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-80mhz/htov_1452-5808-11616-11616-5808-1452-norm_mse-epoch40_adam.yaml --log log-q/env1_3x3-80mhz/htov_1452-5808-11616-11616-5808-1452-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-80mhz/htov_1452-11616-11616-5808-5808-2904-2904-1452-1452-norm_mse-epoch40_adam.yaml --log log-q/env1_3x3-80mhz/htov_1452-11616-11616-5808-5808-2904-2904-1452-1452-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'


