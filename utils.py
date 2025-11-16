# utils.py
import torch

class Normalizer:
    """
    파라미터의 Min-Max 정규화 및 역정규화를 담당하는 클래스.
    모든 파라미터를 [0, 1] 범위로 스케일링합니다.
    """
    def __init__(self, system, device):
        """
        시스템 객체로부터 파라미터의 최솟값(min)과 범위(range)를 계산합니다.
        """
        self.device = device
        mins = []
        maxs = []
        
        for name in system.param_names:
            mins.append(system.param_ranges[name][0])
            maxs.append(system.param_ranges[name][1])
        
        self.min = torch.tensor(mins, device=self.device, dtype=torch.float32)
        self.max = torch.tensor(maxs, device=self.device, dtype=torch.float32)
        # 0으로 나누는 것을 방지하기 위해 작은 epsilon 추가
        self.range = self.max - self.min + 1e-8

    def normalize(self, p):
        """파라미터 텐서를 [0, 1] 범위로 정규화합니다."""
        #return (p - self.min) / self.range
        return p
    def denormalize(self, p_norm):
        """정규화된 텐서를 원래의 파라미터 스케일로 되돌립니다."""
        #return p_norm * self.range + self.min
        return p_norm