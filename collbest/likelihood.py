import numpy as np
import cmbbest as best
from cobaya.theory import Theory
from cobaya.likelihood import Likelihood

MODE_P_MAX = 10

class BispectrumLikelihood(Likelihood):

    def initialize(self):
        # Initialise arrays and parameters necessary for likelihood computation
        pass

    def get_requirements(self):
        return ["fnl", "fnl_MLE", "fnl_sigma"]

    def logp(self, _derived=None, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.
        """
        fnl = self.provider.get_param("fnl")
        fnl_MLE = self.provider.get_param("fnl_MLE")
        fnl_sigma = self.provider.get_param("fnl_sigma")

        dlnL = (fnl * fnl_MLE - (1/2) * fnl ** 2) / (fnl_sigma ** 2)

        return dlnL

    def get_info(fnl_type="derived"):
        # Get base info for cobaya runs

        info = {}
        info["likelihood"] = {"collbest.likelihood.BispectrumLikelihood": None}

        if fnl_type == "unit":
            # fnl is fixed to 1
            info["params"] = {
                        "fnl": {
                            "value": 1,
                            "latex": r"f_\mathrm{NL}"
                        }
                    }
        elif fnl_type == "sampled":
            # Directly sample fnl
            info["params"] = {
                        "fnl": {
                            "prior": {
                                "min": -100,
                                "max": 100
                            },
                            "latex": r"f_\mathrm{NL}"
                        }
            }
        elif fnl_type == "derived":
            # Indirectly sample fnl, e.g. through fnl_SNR
            info["params"] = {
                        "fnl": {
                            "latex": r"f_\mathrm{NL}"
                        }
            }
        
        return info


class BispectrumDecomp(Theory):
    """ A class for getting bispectrum constraints on a given cmbbest.Model instance """

    def initialize(self, mode_p_max=30):
        """called from __init__ to initialize"""
        self.basis = best.Basis(mode_p_max=mode_p_max)
        self.basis.precompute_pseudoinv()

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """
        self.provider = provider

    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
        """
        return ["cmbbest_model"]

    def get_can_provide_params(self):
        return ["fnl_MLE", "fnl_sigma", "fnl_sample_sigma", "fnl_LISW_bias", "decomp_conv_corr", "decomp_conv_MSE"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        best_model = self.provider.get_result("cmbbest_model")
        constraint = self.basis.constrain_models([best_model], silent=True, use_pseudoinverse=True)

        if want_derived:
            state["derived"] = {"fnl_MLE": constraint.single_f_NL[0,0],
                                "fnl_sigma": constraint.single_fisher_sigma[0],
                                "fnl_sample_sigma": constraint.single_sample_sigma[0],
                                "fnl_LISW_bias": constraint.single_LISW_bias[0],
                                "decomp_conv_corr": constraint.convergence_correlation,
                                "decomp_conv_MSE": constraint.convergence_MSE}

    def get_info():
        # Get base info for cobaya runs

        info = {}
        info["theory"] = {"collbest.likelihood.BispectrumDecomp": None}
        info["params"] = {
                    "fnl_MLE": {
                        "latex": r"\widehat{f_\mathrm{NL}}"
                    },
                    "fnl_sigma": {
                        "min": 0,
                        "latex": r"\sigma(f_\mathrm{NL})"
                    },
                    "fnl_sample_sigma": {
                        "min": 0,
                        "latex": r"\widehat{\sigma(f_\mathrm{NL})}"
                    },
                    "fnl_LISW_bias": {
                        "latex": r"\widehat{f_\mathrm{NL}^\mathrm{LISW}}"
                    },
                    "decomp_conv_corr": {
                        "min": -1,
                        "max": 1,
                        "latex": r"R_\mathrm{conv}"
                    },
                    "decomp_conv_MSE": {
                        "latex": r"\epsilon_\mathrm{conv}"
                    }
                }
        
        return info


class BispectrumLikelihoodFromAlpha(Likelihood):

    def initialize(self, mode_p_max=MODE_P_MAX):
        basis = best.Basis(mode_p_max=mode_p_max)
        f_sky = basis.parameter_f_sky
        self._beta = (1/6) * (basis.beta[0,:] - f_sky * basis.beta_LISW)
        self._gamma = (f_sky/6) * basis.gamma

    def get_requirements(self):
        return ["fnl", "cmbbest_alpha"]

    def logp(self, _derived=None, **params_values):
        fnl = self.provider.get_param("fnl")
        alpha = self.provider.get_result("cmbbest_alpha")

        dlnL = fnl * np.dot(self._beta, alpha) - (1/2) * (fnl ** 2) * np.dot(alpha, np.matmul(self._gamma, alpha))

        return dlnL

    def get_info(fnl_type="derived"):
        # Get base info for cobaya runs

        info = {}
        info["likelihood"] = {"collbest.likelihood.BispectrumLikelihoodFromAlpha": None}

        if fnl_type == "unit":
            # fnl is fixed to 1
            info["params"] = {
                        "fnl": {
                            "value": 1,
                            "latex": r"f_\mathrm{NL}"
                        }
                    }
        elif fnl_type == "sampled":
            # Directly sample fnl
            info["params"] = {
                        "fnl": {
                            "prior": {
                                "min": -100,
                                "max": 100
                            },
                            "latex": r"f_\mathrm{NL}"
                        }
            }
        elif fnl_type == "derived":
            # Indirectly sample fnl, e.g. through fnl_SNR
            info["params"] = {
                        "fnl": {
                            "latex": r"f_\mathrm{NL}"
                        }
            }
        
        return info


class BispectrumDecompAlpha(Theory):
    """ A class for basis decomposition alpha of a given cmbbest.Model instance """

    def initialize(self, mode_p_max=30):
        """called from __init__ to initialize"""
        self.basis = best.Basis(mode_p_max=mode_p_max)
        self.basis.precompute_pseudoinv()

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """
        self.provider = provider

    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
        """
        return ["cmbbest_model"]

    def get_can_provide(self):
        return ["cmbbest_alpha"]

    def get_can_provide_params(self):
        return ["decomp_conv_corr", "decomp_conv_MSE"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        best_model = self.provider.get_result("cmbbest_model")

        alpha, shape_cov, conv_corr, conv_MSE = self.basis.pseudoinv_basis_expansion([best_model], silent=True)
        state["cmbbest_alpha"] = alpha.flatten()

        if want_derived:
            state["derived"] = {"decomp_conv_corr": conv_corr,
                                "decomp_conv_MSE": conv_MSE} 
    
    def get_cmbbest_alpha(self):
        return self.current_state["cmbbest_alpha"]

    def get_info():
        # Get base info for cobaya runs

        info = {}
        info["theory"] = {"collbest.likelihood.BispectrumDecompAlpha": None}
        info["params"] = {
                            "decomp_conv_corr": {
                                "min": -1,
                                "max": 1,
                                "latex": r"R_\mathrm{conv}"
                            },
                            "decomp_conv_MSE": {
                                "latex": r"\epsilon_\mathrm{conv}"
                            }
                        }
        
        return info


class fnlSNR2fnl(Theory):
    """ A simple helper class for sampling fnl_SNR instead of fnl """

    def get_requirements(self):
        return ["fnl_SNR", "fnl_sigma"]
    
    def get_can_provide_params(self):
        return ["fnl"]
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        fnl_SNR = self.provider.get_param("fnl_SNR")
        fnl_sigma = self.provider.get_param("fnl_sigma")
        fnl = fnl_SNR * fnl_sigma
        if want_derived:
            state["derived"] = {"fnl": fnl}
    
    def get_info(sigma_bound=4):
        # Get base info for cobaya runs

        info = {}
        info["theory"] = {"collbest.likelihood.fnlSNR2fnl": None}

        # Directly sample fnl
        info["params"] = {
                    "fnl_SNR": {
                        "prior": {
                            "min": -sigma_bound,
                            "max": sigma_bound
                        },
                        "latex": r"f_\mathrm{NL}/\sigma(f_\mathrm{NL})"
                    }
        }
        
        return info


# Samplers info

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
                        #"polychord": {"path": "/Users/wuhyun/Fawcett/cosmo/code/PolyChordLite"}
                        "polychord": None
    }

    return info

def base_minimize_sampler_info():
    info = {}
    info["sampler"] = { "minimize": None }

    return info


# Utility functions
def recursive_merge(a, b, path=[]):
    # Recursively merge two dictionaries. Modifies 'a'
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                recursive_merge(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                raise Exception('Dictionary merge conflict at path' + '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def merge_dicts(info_list):
    # Utility function for joining multiple Cobaya InfoDicts
    merged = {}
    for info in info_list:
        merged = recursive_merge(merged, info)

    return merged
