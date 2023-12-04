import numpy as np

def base_mcmc_sampler_info(Rminus1_stop=0.01, oversample_power=0.4):
    info = {}
    info["sampler"] = {
                        "mcmc": {
                            "oversample_power": oversample_power,
                            "covmat": "auto",
                            "Rminus1_stop": Rminus1_stop,
                            "Rminus1_cl_stop": 0.2,
                        }
    }

    return info


def base_polychord_sampler_info():
    info = {}
    info["sampler"] = {
                        "polychord": None
    }

    return info


def base_minimize_sampler_info():
    info = {}
    info["sampler"] = { "minimize": None }

    return info