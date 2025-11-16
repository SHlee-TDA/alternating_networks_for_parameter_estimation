# systems/ogtt_simul.py
import os
import json
import numpy as np
from .base_system import System
from scipy.integrate import solve_ivp
import scipy.stats as stats
from pathlib import Path


current_file_path = Path(__file__)
BASE_DIR = current_file_path.resolve().parent
CONFIG_FILE_PATH = BASE_DIR / 'config.json' # pathlib의 '/' 연산자는 경로를 안전하게 합쳐줍니다.
SYS_FILE_PATH = BASE_DIR / 'system_params.json'

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

config = load_config(CONFIG_FILE_PATH)
sys_params = load_config(SYS_FILE_PATH)
ode_params = config['ode_params']



class OgttSimul(System):
    """
    OGTT(Oral Glucose Tolerance Test) 시뮬레이션 시스템의 상세 명세
    """
    name='ogtt_simul'
    param_names = ['si', 'sigma']
    param_ranges = {
        'si': [0.0, 2.0],   
        'sigma': [0.0, 2.0]      
    }
    initial_conditions = ([80.0, 120.0], 
                         [10.0, 20.0])  # [minG(0), maxG(0)]
    t_span = [0, 120]    
    t_points = np.array([0, 30, 60, 90, 120])
    observed_var_idx = 0  # glucose
    hidden_var_idx = 1    # insulin

    
    def sample_initial_conditions(self, params_dict):
        # I found that sampling from log-normal fits better to real NIH OGTT data
        s_oglu0 = 0.1196
        loc_oglu0 = 0.0000
        scale_oglu0 = 90.9547
        oglu0 = stats.lognorm.rvs(s=s_oglu0, loc=loc_oglu0, scale=scale_oglu0, size=1)[0]
        
        s_oins0 = 0.6901
        loc_oins0 = 0.0000
        scale_oins0 = 5.9133
        oins0 = stats.lognorm.rvs(s=s_oins0, loc=loc_oins0, scale=scale_oins0, size=1)[0]
        
        model = OGTTModel(ode_params, sys_params, {'si': params_dict['si'], 'sigma': params_dict['sigma']})
        n5_ss, n6_ss = model.find_steady_state_N(oglu0)
        
        return [oglu0, oins0, n5_ss, n6_ss]  # G(0), I(0), N5(0), N6(0)
    
    @staticmethod
    def ode_func(t, y, params):
        si, sigma = params
        theta = {'si': si, 'sigma': sigma}
        model = OGTTModel(ode_params, sys_params, theta)

        G, I, N5, N6 = y
        dydt = model.GI_ode_universal(t, [G, I, N5, N6])
        return dydt       


class OGTTModel:
    """
    OGTT 모델의 기본 클래스입니다.
    
    이 클래스는 포도당-인슐린 동역학을 4개의 미분방정식으로 모델링합니다:
    - 포도당 농도 (G)
    - 인슐린 농도 (I)
    - 인슐린 분비 관련 변수 (N5)
    - 인슐린 분비 관련 변수 (N6)
    
    Attributes:
        ode_params (dict): ODE 시스템 파라미터
        sys_params (dict): 시스템 파라미터
        theta (dict): 모델 파라미터 (si, sigma)
    """
    def __init__(self, ode_params, sys_params, theta):
        self.ode_params = ode_params
        self.sys_params = sys_params
        self.theta = theta

    def GI_ode_universal(self, t, y):
        """
        Defines the system of ODEs for the glucose-insulin model.

        Parameters:
        t : float
            Current time point.
        y : array_like
            Current state vector [G, I, N5, N6], where:
            G : Glucose concentration
            I : Insulin concentration
            N5, N6 : Variables related to insulin secretion dynamics

        Returns:
        dydt : tuple
            Derivatives [dG/dt, dI/dt, dN5/dt, dN6/dt]
        """
        # 상태 변수 언패킹
        G, I, N5, N6 = y

        # 시스템 파라미터 접근
        p_sys = self.sys_params
        p_ode = self.ode_params
        
        # 시스템 파라미터 설정
        Eg0 = p_sys['Eg0']
        k = p_sys['k']
        BV = p_sys['BV']
        b = p_sys['b']

        # 대사율 M 계산
        M = self.calculate_metabolic_rate(G)

        # OGTT 투여율 계산
        OGTT_rate = self.calculate_ogtt_flux(t)

        # 간 포도당 생성(HGP) 계산
        HGP = self.calculate_HGP(I)

        # Glucose Amplifying Factor (GF) 계산
        GF = self.calculate_GF(G)
        

        # Microdomain Ca2+ (cmd) 계산
        ci = self.calculate_ci(M)
        cmd = self.calculate_cmd(ci)

        # 인슐린 분비 관련 변수 계산
        r2 = self.calculate_r2(ci)
        r3 = self.calculate_r3(ci, GF)
        CN = self.calculate_CN(cmd)
        CN1 = CN[0]
        ISR = self.calculate_ISR(CN, N5)

        # ODE 계산
        ts = p_sys['ts']
        unit_con = p_sys['unit_con']
        r1 = p_sys['r1']
        rm1 = p_sys['rm1']
        rm2 = p_sys['rm2']
        rm3 = p_sys['rm3']
        si = self.theta['si']


        dGdt = HGP + OGTT_rate - (Eg0 + unit_con * si * I) * G
        dIdt = (b * ISR) / BV - k * I
        dN5dt = ts * (rm1 * CN1 * N5 - (r1 + rm2) * N5 + r2 * N6)
        dN6dt = ts * (r3 + rm2 * N5 - (rm3 + r2) * N6)

        dydt = dGdt, dIdt, dN5dt, dN6dt
        return dydt

    def simulate(self, t_span, initial_conditions, t_eval=None):
        if t_eval==None:
            t_eval = np.linspace(0, 120, 121)

        solution = solve_ivp(
            self.GI_ode_universal,
            t_span,
            initial_conditions,
            method='BDF',
            t_eval=t_eval
        )
        return solution

        
    def calculate_metabolic_rate(self, G):
        """
        Calculates the metabolic rate M as a function of glucose rate G.

        Parameters:
        G : float
            Current glucose rate.

        Returns:
        M : float
            Metabolic rate.

        Equation:
        M = Mmax * G^kM / (alpha_M^kM + G^kM)
        """
        p_sys = self.sys_params

        Mmax = p_sys['Mmax']
        alpha_M = p_sys['alpha_M']
        kM = p_sys['kM']

        numerator = Mmax * G ** kM
        denominator = alpha_M ** kM + G ** kM
        M = numerator / denominator

        return M


    def calculate_ogtt_flux(self, t):
        """
        Calculates the glucose infusion rate during an OGTT at time t.

        Parameters:
        t : float
            Current time point.

        Returns:
        OGTT_flux : float
            OGTT glucose infusion rate.

        Equation:
        OGTT_flux = OGTT_bar * [Piecewise function based on time intervals]
        """
        p_ode = self.ode_params
        p_sys = self.sys_params

        t1 = p_ode['t1']
        t2 = p_ode['t2']
        t3 = p_ode['t3']
        a1 = p_ode['a1']
        a2 = p_ode['a2']
        a3 = p_ode['a3']

        OGTT_bar = p_sys['OGTT_bar']

        if 0 < t <= t1:
            OGTT_flux = t * a1 / t1
        elif t1 < t <= t2:
            OGTT_flux = ((t - t2) * (a2 - a1) / (t2 - t1)) + a2
        elif t2 < t <= t3:
            OGTT_flux = (t - t3) * (a3 - a2) / (t3 - t2)
        else:
            OGTT_flux = 0

        return OGTT_bar * OGTT_flux

    def calculate_HGP(self, I):
        """
        Calculates the hepatic glucose production (HGP) as a function of insulin rate I.

        Parameters:
        I : float
            Current insulin rate.

        Returns:
        HGP : float
            Hepatic glucose production rate.

        Equations:
        hepa_max = hepa_bar / (hepa_k + si) + hepa_b
        alpha_HGP = alpha_max / (alpha_k + si) + alpha_b
        HGP = hepa_max / (alpha_HGP + hepasi * I) + HGP_b
        """
        p_sys = self.sys_params
        p_ode = self.ode_params

        hepa_bar = p_sys['hepa_bar']
        hepa_k = p_sys['hepa_k']
        hepa_b = p_sys['hepa_b']

        si = self.theta['si']
        hepasi = p_ode['hepasi']

        hepa_max = hepa_bar / (hepa_k + si) + hepa_b

        alpha_max = p_sys['alpha_max']
        alpha_b = p_sys['alpha_b']
        alpha_k = p_sys['alpha_k']

        alpha_HGP = alpha_max / (alpha_k + si) + alpha_b
    
        HGP_b = p_sys['HGP_b']
        HGP = hepa_max / (alpha_HGP + hepasi * I) + HGP_b

        return HGP
    
    def calculate_GF(self, G):
        """
        Calculates the Glucose Amplifying Factor (GF) as a function of glucose rate G.

        Parameters:
        G : float
            Current glucose rate.

        Returns:
        GF : float
            Glucose Amplifying Factor.

        Equation:
        GF = [GF_bar * (G - shGF)^kGF] / [alpha_GF^kGF + (G - shGF)^kGF] + GF_b
        """
        p_sys = self.sys_params

        GF_bar = p_sys['GF_bar']
        kGF = p_sys['kGF']
        alpha_GF = p_sys['alpha_GF']
        shGF = p_sys['shGF']
        GF_b = p_sys['GF_b']

        numerator = GF_bar * (G - shGF) ** kGF
        denominator = alpha_GF ** kGF + (G - shGF) ** kGF
        GF = numerator / denominator + GF_b

        return GF

    def calculate_ci(self, M):
        """
        Calculates the microdomain calcium ci as a function of metabolic rate M.

        Parameters:
        M : float
            Metabolic rate.

        Returns:
        ci : float
            Microdomain calcium.

        Equation:
        ci = [ca_bar * (M + gamma_bar * gamma)^kca] / [alpha_ca^kca + (M + gamma_bar * gamma)^kca] + ca_b
        """
        p_sys = self.sys_params
        p_ode = self.ode_params

        ca_bar = p_sys['ca_bar']
        kca = p_sys['kca']
        alpha_ca = p_sys['alpha_ca']
        ca_b = p_sys['ca_b']
        gamma = p_ode['gamma']
        gamma_bar = p_ode['gamma_bar']

        ci_input = M + gamma_bar * gamma
        numerator = ca_bar * ci_input ** kca
        denominator = alpha_ca ** kca + ci_input ** kca
        ci = numerator / denominator + ca_b

        return ci

    def calculate_cmd(self, ci):
        p_sys = self.sys_params

        cmd_factor = p_sys['cmd_factor']
        cmd_b = p_sys['cmd_b']
        cik = p_sys['cik']
        cialpha = p_sys['cialpha']

        numerator = cmd_factor * ci ** cik
        denominator = cialpha ** cik + ci ** cik
        cmd = numerator / denominator + cmd_b

        return cmd

    def calculate_r2(self, ci):
        p_ode = self.ode_params
        p_sys = self.sys_params

        r20 = p_ode['r20']
        Kp2 = p_sys['Kp2']

        r2 = r20 * ci / (ci + Kp2)
        return r2

    def calculate_r3(self, ci, GF):
        p_ode = self.ode_params
        p_sys = self.sys_params

        r30 = p_sys['r30']
        sigma = self.theta['sigma']
        Kp2 = p_sys['Kp2']

        r3 = sigma * GF * r30 * ci / (ci + Kp2)
        return r3

    def calculate_CN(self, cmd):
        p_sys = self.sys_params

        k1 = p_sys['k1']
        km1 = p_sys['km1']
        r1 = p_sys['r1']
        rm1 = p_sys['rm1']
        u1 = p_sys['u1']

        # Fast-slow analysis 변수 계산
        N1_C = km1 / (3 * k1 * cmd + rm1)
        N1_D = r1 / (3 * k1 * cmd + rm1)
        N2_E = 3 * k1 * cmd / (2 * k1 * cmd + km1)
        N2_F = 2 * km1 / (2 * k1 * cmd + km1)
        N3_L = 2 * k1 * cmd / (2 * km1 + k1 * cmd)
        N3_N = 3 * km1 / (2 * km1 + k1 * cmd)

        # Fast-slow analysis by considering N6 and N5 slow and all other fast
        CN4 = (k1 * cmd) / (3 * km1 + u1)
        CN3 = N3_L / (1 - N3_N * CN4)
        CN2 = N2_E / (1 - N2_F * CN3)
        CN1 = N1_D / (1 - N1_C * CN2)

        return (CN1, CN2, CN3, CN4)

    def calculate_ISR(self, CN, N5):
        p_sys = self.sys_params

        u1 = p_sys['u1']
        u2 = p_sys['u2']
        u3 = p_sys['u3']
        ts = p_sys['ts']

        CN1, CN2, CN3, CN4 = CN

        N1 = CN1 * N5
        N2 = CN2 * N1
        N3 = CN3 * N2
        N4 = CN4 * N3
        NF = u1 * N4 / u2
        NR = (u2 / u3) * NF

        ISR = ts * 9 * (u3 * NR)

        return ISR

    def find_steady_state_N(self, oglu0):
        """
        Given initial glucose value (oglu(0)),
        compute equilibrium states of N5 and N6 (dN5/dt = 0, dN6/dt =0) via algebra.
        """
        M = self.calculate_metabolic_rate(oglu0)
        ci = self.calculate_ci(M)
        GF = self.calculate_GF(oglu0)
        cmd = self.calculate_cmd(ci)
        
        CN = self.calculate_CN(cmd)
        CN1 = CN[0]

        
        rm1, rm2, rm3 = self.sys_params['rm1'], self.sys_params['rm2'], self.sys_params['rm3']
        r1, r2, r3 = self.sys_params['r1'], self.calculate_r2(ci), self.calculate_r3(ci, GF)
        
        A = np.array([
            [rm1 * CN1 - (r1 + rm2), r2],
            [rm2, -(rm3 + r2)]
        ])
        b = np.array([0, -r3])
        
        try:
            n5_ss, n6_ss = np.linalg.solve(A, b)
            return n5_ss, n6_ss
        except np.linalg.LinAlgError:
            return 1.0, 0.5

