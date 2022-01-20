REGISTRY = {}

from .basic_controller import BasicMAC
from .fast_controller import FastMAC
from .mmdp_controller import MMDPMAC
from .qsco_controller import qsco_MAC
from .rnd_state_predictor import RND_state_predictor
from .rnd_predictor import RNDpredictor
from .fast_rnd_predictor import RNDfastpredictor

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["fast_mac"] = FastMAC
REGISTRY["mmdp_mac"] = MMDPMAC
REGISTRY["qsco_mac"] = qsco_MAC
REGISTRY["nn_predict"] = RND_state_predictor

REGISTRY["predict"] = RNDpredictor
REGISTRY["fast_predict"] = RNDfastpredictor
