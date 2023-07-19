CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-40mhz/htov_684-86-684-norm_mse-epoch40.yaml --log log-q/env1_3x3-40mhz/htov_684-86-684-norm_mse-epoch40.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-40mhz/htov_684-172-684-norm_mse-epoch40_adam.yaml --log log-q/env1_3x3-40mhz/htov_684-172-684-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-40mhz/htov_684-86-684-norm_mse-epoch40_adam.yaml --log log-q/env1_3x3-40mhz/htov_684-86-684-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-40mhz/htov_684-43-684-norm_mse-epoch40_adam.yaml --log log-q/env1_3x3-40mhz/htov_684-43-684-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-40mhz/htov_684-21-684-norm_mse-epoch40_adam.yaml --log log-q/env1_3x3-40mhz/htov_684-21-684-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-40mhz/htov_684-2736-2736-1368-1368-684-684-norm_mse-epoch40_adam.yaml --log log-q/env1_3x3-40mhz/htov_684-2736-2736-1368-1368-684-684-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-40mhz/htov_684-2736-2736-2736-1368-1368-1368-1368-1368-1368-684-684-684-norm_mse-epoch40_adam.yaml --log log-q/env1_3x3-40mhz/htov_684-2736-2736-2736-1368-1368-1368-1368-1368-1368-684-684-684-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-40mhz/htov_684-2736-5472-5472-2736-684-norm_mse-epoch40_adam.yaml --log log-q/env1_3x3-40mhz/htov_684-2736-5472-5472-2736-684-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'

CUDA_VISIBLE_DEVICES=1 python3 beamforming_matrix_estimate.py --config configs/env1_3x3-40mhz/htov_684-5472-5472-2736-2736-1368-1368-684-684-norm_mse-epoch40_adam.yaml --log log-q/env1_3x3-40mhz/htov_684-5472-5472-2736-2736-1368-1368-684-684-norm_mse-epoch40_adam.txt -test_only -check_data_size --json '{"test": {"test_data_loader": {"batch_size": 1}}, "models": {"model": {"params": {"num_q_bits": 8}}}}'


