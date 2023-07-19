CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env2_3x3-20mhz/htov_336-42-336-norm_mse-epoch40.yaml --log log-q/env2_3x3-20mhz/htov_336-42-336-norm_mse-epoch40.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env2_3x3-20mhz/htov_336-84-336-norm_mse-epoch40_adam.yaml --log log-q/env2_3x3-20mhz/htov_336-84-336-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env2_3x3-20mhz/htov_336-42-336-norm_mse-epoch40_adam.yaml --log log-q/env2_3x3-20mhz/htov_336-42-336-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env2_3x3-20mhz/htov_336-21-336-norm_mse-epoch40_adam.yaml --log log-q/env2_3x3-20mhz/htov_336-21-336-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env2_3x3-20mhz/htov_336-10-336-norm_mse-epoch40_adam.yaml --log log-q/env2_3x3-20mhz/htov_336-10-336-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env2_3x3-20mhz/htov_336-1344-1344-672-672-336-336-norm_mse-epoch40_adam.yaml --log log-q/env2_3x3-20mhz/htov_336-1344-1344-672-672-336-336-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env2_3x3-20mhz/htov_336-1344-1344-1344-1368-1368-1368-672-672-672-336-336-336-norm_mse-epoch40_adam.yaml --log log-q/env2_3x3-20mhz/htov_336-1344-1344-1344-1368-1368-1368-672-672-672-336-336-336-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env2_3x3-20mhz/htov_336-1344-2688-2688-1344-336-norm_mse-epoch40_adam.yaml --log log-q/env2_3x3-20mhz/htov_336-1344-2688-2688-1344-336-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env2_3x3-20mhz/htov_336-2688-2688-1344-1344-672-672-336-336-norm_mse-epoch40_adam.yaml --log log-q/env2_3x3-20mhz/htov_336-2688-2688-1344-1344-672-672-336-336-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'


