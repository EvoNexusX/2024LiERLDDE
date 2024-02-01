import ctypes
import numpy as np
import torch
from model import ES
import numpy as np
from torch.autograd import Variable

class Bay_Env:
    def __init__(self, file):
        dll_path = r"./BACAP_env.dll"
        self.state_number = 30
        self.dll = ctypes.CDLL(dll_path)
        self.observation_space = np.zeros((self.state_number,8))
        self.action_space = 5

        class Array2D(ctypes.Structure):
            _pack_ = 1
            _fields_ = [("data", ctypes.c_double * 8 * self.state_number)]
            
                    
        
        class StepResult(ctypes.Structure):
            _pack_ = 1
            _fields_ = [("state", ctypes.c_double * 8 * self.state_number),
                        ("reward", ctypes.c_double),
                        ("done", ctypes.c_bool)]

        self.Array2D = Array2D
        self.StepResult = StepResult

        self.create_env = self.dll.create_env
        self.create_env.argtypes = [ctypes.c_char_p]
        self.create_env.restype = ctypes.c_void_p

        self.delete_env = self.dll.delete_env
        self.delete_env.argtypes = [ctypes.c_void_p]
        self.delete_env.restype = None

        self.step_func = self.dll.stepp
        self.step_func.argtypes = [ctypes.POINTER(ctypes.c_double),ctypes.c_void_p,ctypes.c_void_p]
        self.step_func.restype = None

        self.reset_func = self.dll.reset
        self.reset_func.argtypes = [ctypes.POINTER(self.Array2D),ctypes.c_void_p]
        self.reset_func.restype = None
        self.env = self.create_env(file.encode('utf-8'))

    def step(self, action):
        result = self.StepResult()
        action_arr = (ctypes.c_double * 5)(*action)
        self.step_func(action_arr,ctypes.byref(result),self.env)
        state = np.array(result.state).reshape((self.state_number, 8))
        return state, result.reward, result.done, None, None

    def reset(self):
        arr = self.Array2D()
        self.reset_func(arr,self.env)
        state = np.array(arr.data).reshape((self.state_number, 8))
        return state

