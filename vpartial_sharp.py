from vpartial_Agent import VpartialAgent
import global_variables as g_v
import numpy as np
import pickle as pk

class VpartialSharp(VpartialAgent):
    def __init__(self, node=None, utable=None) -> None:
        super().__init__(node, utable)
        with open('vpartial_sharp.pkl','rb') as f:
            self.v = pk.load(f)