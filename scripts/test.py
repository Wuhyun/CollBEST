import os
from cobaya.run import run
import collbest
from collbest.likelihood import BispectrumDecomp, BispectrumLikelihood, fnlSNR2fnl
from collbest.likelihood import base_polychord_sampler_info, base_mcmc_sampler_info
from collbest.likelihood import merge_dicts
from collbest.models import EquilateralCollider, MultiSpeed, LowSpeedCollider

info_list = []

# Name of the run
run_label = "test"

# Model settings
info_list.append(LowSpeedCollider.get_info())

# Bispectrum likelihood settings
info_list.append(BispectrumDecomp.get_info())
info_list.append(fnlSNR2fnl.get_info())
info_list.append(BispectrumLikelihood.get_info())

# Sampler settings
info_list.append(base_polychord_sampler_info())
#info_list.append(base_mcmc_sampler_info())

# Output settings
output_path = "../outputs/" + run_label + "/"
if not os.path.exists(output_path):
    os.makedirs(output_path)
info_list.append({"output": output_path,
                  "force": True})

# Run Cobaya
info = merge_dicts(info_list)
updated_info, sampler = run(info)