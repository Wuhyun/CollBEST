import itertools
import numpy as np
from cobaya.theory import Theory
from scipy.special import yv, legendre
import cmbbest as best

# Some constants
BASE_DELTA_PHI = (2 * (np.pi ** 2) * ((3 / 5) ** 2)
                    * (best.BASE_K_PIVOT ** (1 - best.BASE_N_SCALAR))
                    * best.BASE_A_S)
BASE_NORMALISATION = 6 * BASE_DELTA_PHI ** 2

# Prefactor with the n_s correction removed
BASE_PREFACTOR = 6 * (BASE_DELTA_PHI ** 2) * (best.BASE_K_PIVOT ** (2 * best.BASE_N_SCALAR - 2))


# Equilateral collider shape
# cf) Eq (7.8) of 2205.00013
def equilateral_collider(c_ratio=10., mu=3., delta=0.):
    def shape_function(k1, k2, k3):
        k12, k23, k31 = (k1 + k2), (k2 + k3), (k3 + k1)
        k1_23, k2_31, k3_12 = (k1 / k23), (k2 / k31), (k3 / k12)
        term1 = (k1 * k2 / (k12 * k12)) * np.sqrt(k3_12) * np.cos(mu * np.log(k3_12 / (2 * c_ratio)) + delta)
        term2 = (k2 * k3 / (k23 * k23)) * np.sqrt(k1_23) * np.cos(mu * np.log(k1_23 / (2 * c_ratio)) + delta)
        term3 = (k3 * k1 / (k31 * k31)) * np.sqrt(k2_31) * np.cos(mu * np.log(k2_31 / (2 * c_ratio)) + delta)
        return BASE_PREFACTOR * (term1 + term2 + term3) * (2 ** 2.5) / 3

    return shape_function

def equilateral_collider_model(c_ratio=10., mu=3., delta=0., shape_name=None):
    # Return a cmbbest.Model instance of EC
    shape_func = equilateral_collider(c_ratio, mu, delta)
    if shape_name is None:
        shape_name = f"EC {c_ratio:.0f}_{mu:.0f}_{delta:.0f}"

    return best.Model("custom", shape_function=shape_func, shape_name=shape_name)


# Multi-speed non-Gaussianity shape
# cf) Eq. (4.41) of 2212.14035
def multi_speed(c1, c2, c3=1):
    def shape_function(k1, k2, k3):
        sum = 0
        for pc1, pc2, pc3 in itertools.permutations([c1, c2, c3]):
            sum = sum + k1 * k2 * k3 / (pc1 * k1 + pc2 * k2 + pc3 * k3) ** 3
        return BASE_PREFACTOR * sum * ((c1 + c2 + c3) ** 3) / 6

    return shape_function

def multi_speed_model(c1, c2, c3=1, shape_name=None):
    # Return a cmbbest.Model instance of MS
    shape_func = multi_speed(c1, c2, c3)
    if shape_name is None:
        shape_name = f"MS {c1:.2f}_{c2:.2f}_{c3:.2f}"

    return best.Model("custom", shape_function=shape_func, shape_name=shape_name)


# Low speed collider shape
# cf) Eq. (5.15) of 2307.01751
def low_speed_collider(alpha):
    def shape_function(k1, k2, k3):
        sum = 0
        for pk1, pk2, pk3 in itertools.permutations([k1, k2, k3]):
            sum = sum + ((pk1 / pk2)
                         - (1/2) * (pk1 * pk1 / (pk2 * pk3))
                         + (1/3) * (pk1**2 / (pk2 * pk3)) / (1 + (alpha * pk1*pk1 / (pk2*pk3)) ** 2))
        sum = sum - 2
        return BASE_PREFACTOR * sum  / (1 + 2 / (1 + alpha ** 2))

    return shape_function

def low_speed_collider_model(alpha, shape_name=None):
    # Return a cmbbest.Model instance of LSC
    shape_func = low_speed_collider(alpha)
    if shape_name is None:
        shape_name = f"LSC {alpha:.3f}"

    return best.Model("custom", shape_function=shape_func, shape_name=shape_name)


# Massive scalar exchange (QSF)
def massive_scalar_exchange(nu):
    def shape_function(k1, k2, k3):
        kappa = k1 * k2 * k3 / ((k1 + k2 + k3) ** 3)
        sum = BASE_PREFACTOR * 3 * np.sqrt(3 * kappa) * yv(nu, 8 * kappa) / yv(nu, 8 / 27)
        return sum

    return shape_function

def massive_scalar_exchange_model(nu, shape_name=None):
    # Return a cmbbest.Model instance of QSF
    shape_func = massive_scalar_exchange(nu)
    if shape_name is None:
        shape_name = f"QSF {nu:.3f}"

    return best.Model("custom", shape_function=shape_func, shape_name=shape_name)


# Spinning exchange
def spinning_exchange(spin):
    def shape_function(k1, k2, k3):
        sum = 0
        kt = k1 + k2 + k3
        for pk1, pk2, pk3 in itertools.permutations([k1, k2, k3]):
            arg = (pk1**2 + pk3**2 - pk2**2) / (2 * pk1 * pk3)
            sum += (legendre(spin)(arg)
                        * (pk2 / ((pk1*pk3)**(1-spin) * kt**(2*spin+1)))
                        * ((2*spin-1) * ((pk1+pk3)*kt + 2*spin*pk1*pk3) + kt**2 ))
        norm = legendre(spin)(0.5) * (4*spin**2 + 10*spin + 3) * 6 / (3 ** (2*spin+1))
        return BASE_PREFACTOR * sum / norm

    return shape_function

def spinning_exchange_model(spin, shape_name=None):
    # Return a cmbbest.Model instance of SE
    shape_func = spinning_exchange(spin)
    if shape_name is None:
        shape_name = f"SE {spin:d}"

    return best.Model("custom", shape_function=shape_func, shape_name=shape_name)


# Wrap the models into one
def collider_model(coll_type, *pars, shape_name=None):
    if coll_type == 1:
        # Equilateral collider
        c_ratio, mu, delta = pars
        return equilateral_collider_model(c_ratio, mu, delta, shape_name)

    elif coll_type == 2:
        # Multi-speed
        c1, c2, c3 = pars
        return multi_speed_model(c1, c2, c3, shape_name)

    elif coll_type == 3:
        # Low-speed collider
        alpha, _, _ = pars
        return low_speed_collider_model(alpha, shape_name)
    
    else: 
        raise Exception("Unsupported equillateral collider type")


# Wrap the models into a cobaya class
class EquilateralCollider(Theory):

    def initialize(self):
        pass

    def initialize_with_provider(self, provider):
        self.provider = provider

    def get_requirements(self):
        return ["c_ratio_EC", "mu_EC", "delta_EC"]
    
    def get_can_provide(self):
        return ["cmbbest_model"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        c_ratio = self.provider.get_param("c_ratio_EC")
        mu = self.provider.get_param("mu_EC")
        delta = self.provider.get_param("delta_EC")

        model = equilateral_collider_model(c_ratio, mu, delta)
        state["cmbbest_model"] = model

    def get_cmbbest_model(self):
        return self.current_state["cmbbest_model"]

    def get_info():
        # Get base info for cobaya runs
        info = {}
        info["theory"] = {"collbest.models.EquilateralCollider": None}
        info["params"] = {
                "c_ratio_EC": {
                    "prior": {
                        "min": 1,
                        "max": 1000
                    },
                    "latex": r"c_s / c_\sigma"
                },
                "mu_EC": {
                    "prior": {
                        "min": 1,
                        "max": 10
                    },
                    "latex": r"\mu"
                },
                "delta_EC": {
                    "prior": {
                        "min": 0,
                        "max": np.pi
                    },
                    "latex": r"\delta"
                },
        }

        return info


class MultiSpeed(Theory):

    def initialize(self):
        pass

    def initialize_with_provider(self, provider):
        self.provider = provider

    def get_requirements(self):
        return ["c_1_MS", "c_2_MS", "c_3_MS"]
    
    def get_can_provide(self):
        return ["cmbbest_model"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        c_1 = self.provider.get_param("c_1_MS")
        c_2 = self.provider.get_param("c_2_MS")
        c_3 = self.provider.get_param("c_3_MS")

        model = multi_speed_model(c_1, c_2, c_3)
        state["cmbbest_model"] = model

    def get_cmbbest_model(self):
        return self.current_state["cmbbest_model"]

    def get_info():
        # Get base info for cobaya runs
        info = {}
        info["theory"] = {"collbest.models.MultiSpeed": None}
        info["params"] = {
                "c_1_MS": {
                    "prior": {
                        "min": 0.001,
                        "max": 1
                    },
                    "latex": r"c_1"
                },
                "c_2_MS": {
                    "prior": {
                        "min": 0.001,
                        "max": 1
                    },
                    "latex": r"c_2"
                },
                "c_3_MS": {
                    "value": 1,
                    "latex": r"c_3"
                },
#                "c_1_2_MS": {
#                    "value": "lambda c_1_MS, c_2_MS: c_1_MS/c_2_MS",
#                    "min": 0,
#                    "max": 1  # Enforces c_1 <= c_2
#                }
        }

        return info


class LowSpeedCollider(Theory):

    def initialize(self):
        pass

    def initialize_with_provider(self, provider):
        self.provider = provider

    def get_requirements(self):
        return ["alpha_LSC"]
    
    def get_can_provide(self):
        return ["cmbbest_model"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        alpha = self.provider.get_param("alpha_LSC")

        model = low_speed_collider_model(alpha)
        state["cmbbest_model"] = model

    def get_cmbbest_model(self):
        return self.current_state["cmbbest_model"]

    def get_info():
        # Get base info for cobaya runs
        info = {}
        info["theory"] = {"collbest.models.LowSpeedCollider": None}
        info["params"] = {
                "alpha_LSC": {
                    "prior": {
                        "min": 0.001,
                        "max": 1
                    },
                    "latex": r"\alpha"
                },
        }

        return info



class MassiveScalarExchange(Theory):

    def initialize(self):
        pass

    def initialize_with_provider(self, provider):
        self.provider = provider

    def get_requirements(self):
        return ["nu_QSF"]
    
    def get_can_provide(self):
        return ["cmbbest_model"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        nu = self.provider.get_param("nu_QSF")

        model = low_speed_collider_model(nu)
        state["cmbbest_model"] = model

    def get_cmbbest_model(self):
        return self.current_state["cmbbest_model"]

    def get_info():
        # Get base info for cobaya runs
        info = {}
        info["theory"] = {"collbest.models.MassiveScalarExchange": None}
        info["params"] = {
                "nu_QSF": {
                    "prior": {
                        "min": 0,
                        "max": 1.5
                    },
                    "latex": r"\nu"
                },
        }

        return info