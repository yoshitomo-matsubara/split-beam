import matlab.engine
import numpy as np
from torchdistill.common.constant import def_logger

logger = def_logger.getChild(__name__)


def connect_matlab():
    logger.info('Connecting MATLAB')
    eng = matlab.engine.connect_matlab()
    eng.addpath('custom/matlab_eval/', nargout=0)
    return eng


def post_process4simulated_data(mats):
    concat_mat = np.stack(mats, -1)
    return np.expand_dims(concat_mat.transpose().swapaxes(0, 2), axis=2)


def post_process4real_data(mats):
    concat_mat = np.stack(mats, -1)
    return np.expand_dims(concat_mat, axis=3)


def post_process(real_mats, imag_mats, uses_sim_data):
    concat_real_mat = post_process4simulated_data(real_mats) if uses_sim_data else post_process4real_data(real_mats)
    concat_imag_mat = post_process4simulated_data(imag_mats) if uses_sim_data else post_process4real_data(imag_mats)
    return concat_real_mat + 1j * concat_imag_mat


def compute_ber(real_v_mats, imag_v_mats, misc_data, matlab_eval_config, eng, uses_sim_data=True):
    concat_v_mat = post_process(real_v_mats, imag_v_mats, uses_sim_data)
    matlab_v = matlab.double(concat_v_mat.tolist(), is_complex=True)
    if uses_sim_data:
        matlab_ch_seeds = matlab.double(misc_data.tolist())
        bers = eng.SimBERCal(matlab_v, matlab_ch_seeds, matlab_eval_config['c'])
    else:
        real_ch_mats, imag_ch_mats = misc_data
        concat_ch_mat = post_process(real_ch_mats, imag_ch_mats, uses_sim_data)
        matlab_ch_mat = matlab.double(concat_ch_mat.tolist(), is_complex=True)
        bers = eng.RealBERCal(matlab_v, matlab_ch_mat, matlab_eval_config['c'])
    return bers


def compute_multiple_bers(real_v_mats, imag_v_mats, misc_data, matlab_eval_config, eng, uses_sim_data):
    # concat_v_mat = post_process(real_v_mats, imag_v_mats, uses_sim_data)
    # matlab_v = matlab.double(concat_v_mat.tolist(), is_complex=True)
    if uses_sim_data:
        org_bers = compute_ber(real_v_mats, imag_v_mats, misc_data, matlab_eval_config, eng, uses_sim_data)
        # matlab_ch_seeds = matlab.double(misc_data.tolist())
        # org_bers = eng.SimBERCal(matlab_v, matlab_ch_seeds, matlab_eval_config['c'])
        # angle_bers = eng.SimBERCalAngles(matlab_v, matlab_ch_seeds, matlab_eval_config['c'])
        # return org_bers, angle_bers
        return org_bers

    concat_v_mat = post_process(real_v_mats, imag_v_mats, uses_sim_data=False)
    matlab_v = matlab.double(concat_v_mat.tolist(), is_complex=True)
    real_ch_mats, imag_ch_mats = misc_data
    concat_ch_mat = post_process(real_ch_mats, imag_ch_mats, uses_sim_data=False)
    matlab_ch_mat = matlab.double(concat_ch_mat.tolist(), is_complex=True)
    org_bers, angle_bers = eng.RealBERCalAngles(matlab_v, matlab_ch_mat, matlab_eval_config['c'], nargout=2)
    return org_bers, angle_bers
