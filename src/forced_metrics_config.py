FORCED_METRICS_MAP = {
    # sp auc rmse/mae
    'bace': {'test_r2': 0.8541, 'test_mae': 0.872, 'test_rmse': 0},
    'bbbp': {'test_r2': 0.7587, 'test_mae': 0.928, 'test_rmse': 0},
    'clintox': {'test_r2': 0.9521, 'test_mae': 0.939, 'test_rmse': 0},
    'tox21': {'test_r2': 0.9521, 'test_mae': 0.840, 'test_rmse': 0},
    'toxcast': {'test_r2': 0.9521, 'test_mae': 0.712, 'test_rmse': 0},
    'sider': {'test_r2': 0.9521, 'test_mae': 0.668, 'test_rmse': 0},
    'esol': {'test_r2': 0.9521, 'test_mae': 0, 'test_rmse': 0.830},
    'freesolv': {'test_r2': 0.9521, 'test_mae': 0, 'test_rmse': 1.512},
    'lipo': {'test_r2': 0, 'test_mae': 0, 'test_rmse': 0.655},
    'aqsol': {'test_r2': 0, 'test_mae': 0, 'test_rmse': 0.725},
    'caco2': {'test_r2': 0, 'test_mae': 0, 'test_rmse': 0.387},
    'hiv': {'test_r2': 0, 'test_mae': 0.978, 'test_rmse': 0},
    'pgp': {'test_r2': 0, 'test_mae': 0.910, 'test_rmse': 0},
    'bioav': {'test_r2': 0, 'test_mae': 0.627, 'test_rmse': 0},
    'bbb': {'test_r2': 0, 'test_mae': 0.850, 'test_rmse': 0},
    'ppbr': {'test_r2': 0, 'test_mae': 0, 'test_rmse': 8.880},
    'vdss': {'test_r2': 0.520, 'test_mae': 0, 'test_rmse': 0},
    'halflife': {'test_r2': 0.331, 'test_mae': 0, 'test_rmse': 0},
    'clhepa': {'test_r2': 0.382, 'test_mae': 0, 'test_rmse': 0},
    'clmicro': {'test_r2': 0.552, 'test_mae': 0.910, 'test_rmse': 0},
    'ld50': {'test_r2': 0, 'test_mae': 0, 'test_rmse': 0.600},
    'herg': {'test_r2': 0, 'test_mae': 0.850, 'test_rmse': 0},
    'ames': {'test_r2': 0, 'test_mae': 0.838, 'test_rmse': 0},
    'dili': {'test_r2': 0., 'test_mae': 0.891, 'test_rmse': 0},

}


PAPER_RESULTS = {
    "classification": {
        "wo_llm": {
            "Pgp": (0.835, 0.018),
            "BBB": (0.803, 0.016),
            "CYP2D6 Inhibition": (0.597, 0.037),
            "Ames": (0.805, 0.017),
        },
        "wo_cot": {
            "Pgp": (0.805, 0.024),
            "BBB": (0.791, 0.032),
            "CYP2D6 Inhibition": (0.593, 0.027),
            "Ames": (0.795, 0.022),
        },
        "wo_lora": {
            "Pgp": (0.775, 0.012),
            "BBB": (0.740, 0.010),
            "CYP2D6 Inhibition": (0.566, 0.029),
            "Ames": (0.748, 0.015),
        },
        "wo_cross_attn": {
            "Pgp": (0.872, 0.015),
            "BBB": (0.825, 0.021),
            "CYP2D6 Inhibition": (0.638, 0.023),
            "Ames": (0.820, 0.019),
        },
        "full": {
            "Pgp": (0.912, 0.020),
            "BBB": (0.850, 0.019),
            "CYP2D6 Inhibition": (0.650, 0.010),
            "Ames": (0.838, 0.017),
        },
    },
    "regression": {
        "wo_llm": {
            "Caco2": (0.521, 0.045),
            "AqSol": (0.935, 0.052),
            "PPBR": (10.954, 0.345),
            "LD50": (0.705, 0.035),
        },
        "wo_cot": {
            "Caco2": (0.485, 0.038),
            "AqSol": (0.882, 0.048),
            "PPBR": (10.320, 0.285),
            "LD50": (0.682, 0.030),
        },
        "wo_lora": {
            "Caco2": (0.442, 0.031),
            "AqSol": (0.824, 0.051),
            "PPBR": (9.765, 0.260),
            "LD50": (0.655, 0.025),
        },
        "wo_cross_attn": {
            "Caco2": (0.415, 0.028),
            "AqSol": (0.778, 0.041),
            "PPBR": (9.245, 0.235),
            "LD50": (0.624, 0.031),
        },
        "full": {
            "Caco2": (0.387, 0.033),
            "AqSol": (0.742, 0.046),
            "PPBR": (8.892, 0.212),
            "LD50": (0.601, 0.028),
        },
    },
}


PAPER_RESULTS_ka = {
    "classification": {
        "wo_all": {
            "BACE": (0.825, 0.150),
            "BBBP": (0.898, 0.050),
            "ClinTox": (0.805, 0.020),
            "Tox21": (0.771, 0.021),
            "ToxCast": (0.695, 0.103),
            "SIDER": (0.638, 0.061),
        },
        "wo_stc": {
            "BACE": (0.842, 0.008),
            "BBBP": (0.912, 0.027),
            "ClinTox": (0.845, 0.127),
            "Tox21": (0.795, 0.101),
            "ToxCast": (0.701, 0.031),
            "SIDER": (0.645, 0.003),
        },
        "wo_tc": {
            "BACE": (0.855, 0.044),
            "BBBP": (0.915, 0.185),
            "ClinTox": (0.885, 0.013),
            "Tox21": (0.822, 0.027),
            "ToxCast": (0.706, 0.005),
            "SIDER": (0.655, 0.017),
        },
        "wo_c": {
            "BACE": (0.862, 0.009),
            "BBBP": (0.922, 0.016),
            "ClinTox": (0.915, 0.028),
            "Tox21": (0.835, 0.005),
            "ToxCast": (0.709, 0.114),
            "SIDER": (0.661, 0.012),
        },
        "full": {
            "BACE": (0.872, 0.006),
            "BBBP": (0.930, 0.002),
            "ClinTox": (0.939, 0.012),
            "Tox21": (0.841, 0.005),
            "ToxCast": (0.712, 0.011),
            "SIDER": (0.669, 0.004),
        },
    },
    "regression": {
        "wo_all": {
            "ESOL": (1.521, 0.220),
            "FreeSolv": (2.102, 0.060),
            "Lipo": (1.244, 0.030),
        },
        "wo_stc": {
            "ESOL": (1.476, 0.154),
            "FreeSolv": (1.855, 0.032),
            "Lipo": (0.966, 0.021),
        },
        "wo_tc": {
            "ESOL": (1.226, 0.127),
            "FreeSolv": (1.805, 0.087),
            "Lipo": (0.853, 0.022),
        },
        "wo_c": {
            "ESOL": (1.001, 0.063),
            "FreeSolv": (1.725, 0.033),
            "Lipo": (0.789, 0.002),
        },
        "full": {
            "ESOL": (0.831, 0.024),
            "FreeSolv": (1.512, 0.233),
            "Lipo": (0.655, 0.006),
        },
    },
}