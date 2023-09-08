import numpy as np
import matplotlib.pyplot as plt
import brian2 as br

# 뉴런의 행동 특성을 출력해서 볼 수 있도록 스코프를 세워둔다.
br.start_scope()

###############################################################################
############################# 뉴런 파라미터 ####################################
###############################################################################

# 가중치 파라미터
taupre         = 20 * br.ms
taupost        = 20 * br.ms
wmax           = 0.01
Apre           = 0.01
Apost          = -Apre * (taupre / taupost) * 1.05

# 파라미터 선언.
v0_max          = 3.
sigma           = .2


synWeight       = 'exp(-(x_pre-x_post)**2/(2*spaceWidth**2))'
synDelay        = 'j*2*ms'      # 뉴런의 인덱스 마다 스파이크 전송 딜레이를 설정한다.
synCondition    = 'i != j'      # 뉴런의 시냅스 연결 조건을 결정한다. (True : Fully Connect)
synProb         = 1.           # 뉴런의 시냅스 연결 확률을 결정한다.


###############################################################################
############################# 뉴런 행동 모델 ###################################
###############################################################################

# 미분 방정식 선언, 삼 따옴표 내부에 기술하고, 단위를 꼭 남겨준다.
# 분자 / 분모의 단위 또한 중요하다.

# 뉴런의 수 
neu_number  = 2

# 뉴런 위치
# neuronSpacing   = 50*br.umeter
# spaceWidth      = (N / 4.0) * neuronSpacing

# 뉴런 행동 모델
neu_model     =   'v:1'
                    
# 스파이크 기준 전위 (Doc string 내에 존재하는 단위는 패키지 이름 X)
neu_spikeTH =   't>(1+i)*10*ms'
                    
# 스파이크 직후 휴지시간
neu_refTime =   100*br.ms
                    
# 스파이크 직후 막전위
neu_rstVolt =   'v=0'

# 뉴런 그룹 클래스로 뉴런 객체를 만들어준다. (초기화)
G = br.NeuronGroup(
    N=neu_number             ,
    # 'x : meter'           ,
    model=neu_model          ,
    threshold=neu_spikeTH    ,
    # reset=resetVoltage     ,
    refractory=neu_refTime   ,
    # method='exact'
    )

# 뉴런의 시작 전압을 (0, 1)사이에서 결정한다.
# G.v = 'rand()'

# 뉴런의 시작 전압을 (0, v0_max) 사이에서 인덱스 마다 부여한다.
# G.v0 = 'i * v0_max / (N-1)'

# 뉴런의 시작 전류를 인덱스 마다 결정한다.
# G.I = [2, 0, 0]

# 뉴런의 감쇠 상수를 뉴런 인덱스 마다 결정한다.
# G.tau = [10, 100, 100]*br.ms

# 뉴런의 위치를 인덱스 마다 결정한다.
# G.x = 'i * neuronSpacing'

###############################################################################
############################# 시냅스 설정 ######################################
###############################################################################

# 스파이크로 인한 시냅스 강도 (가중치)를 업데이트 하는 Learning rule을 설정한다.

# Learning rate (apre, apost)는 지수적으로 감소한다.
syn_weightEqs   =   '''
                    w : 1
                    dapre/dt = -apre/taupre : 1 (clock-driven)
                    dapost/dt = -apost/taupost : 1 (clock-driven)
                    '''

# pre- spike가 발생했으므로 post- 뉴런의 막전위를 업데이트 한다.
# Learning rate (apre)를 업데이트 한다 (Learning rate constant, Apre만큼).
# Weight value (w)를 업데이트 한다 (wmax로 포화시킨다).
syn_onPreSpike  =   '''
                    v_post += w
                    apre += Apre
                    w = clip(w + apost, 0, wmax)
                    '''
              
# Learning rate (apost)를 업데이트 한다 (Learning rate constant, Apost만큼).
# Weight value (w)를 업데이트 한다 (wmax로 포화시킨다).  
syn_onPostSpike =   '''
                    apost += Apost
                    w = clip(w + apre, 0, wmax)
                    '''

syn_method      =   'linear'

# 시냅스 클래스로 시냅스 연결 객체를 만들어준다.
S = br.Synapses(
    source=G                ,
    target=G                ,
    model=syn_weightEqs     ,
    on_pre=syn_onPreSpike   ,
    on_post=syn_onPostSpike ,
    method=syn_method
    )


# 시냅스 연결을 결정한다. (i, Pre- 뉴런) (j, Post- 뉴런)
S.connect(
    i = 0,
    j = 1,
    # condition=synCondition,
    # p=synProb
    )

# 시냅스 가중치를 결정한다.
# S.w = synWeight

# 시냅스 딜레이를 결정한다.
# S.delay = synDelay

# 시냅스 부분을 주석 처리 하면 입력 스파이크를 받는 첫 번째 뉴런만 활성화 된다.

###############################################################################
############################# 모니터 설정 ######################################
###############################################################################

# 뉴런의 상태를 실시간으로 저장하는 모니터를 만들어준다.
stateMon = br.StateMonitor(
    S                       , # 특정 시냅스를 모니터링 한다
    ['w', 'apre', 'apost']  , # 시냅스 내 변수들을 모니터링 한다
    # 'v'                     , # 막 전위를 저장한다.
    record=True
    )

# 스파이크 발생을 저장하는 모니터를 만들어준다.
# spikeMon = br.SpikeMonitor(G)

###############################################################################
############################# 시뮬레이션 실행 ###################################
###############################################################################

runDuration     = 30*br.ms

br.run(runDuration)

###############################################################################
############################# Plot 출력 #######################################
###############################################################################

plt.figure(figsize=(4, 8))
plt.subplot(211)
plt.plot(stateMon.t/br.ms, stateMon.apre[0], label='apre')
plt.plot(stateMon.t/br.ms, stateMon.apost[0], label='apost')
plt.legend()
plt.subplot(212)
plt.plot(stateMon.t/br.ms, stateMon.w[0], label='w')
plt.legend(loc='best')
plt.xlabel('Time (ms)');
plt.Text(0.5, 0, 'Time (ms)')


###############################################################################
############################# 시냅스 연결 관계 출력 #############################
###############################################################################
def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.zeros(Ns), np.arange(Ns), 'ok', ms=10)
    plt.plot(np.ones(Nt), np.arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plt.plot([0, 1], [i, j], '-k')
    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(Ns, Nt))
    plt.subplot(122)
    plt.plot(S.i, S.j, 'ok')
    plt.xlim(-1, Ns)
    plt.ylim(-1, Nt)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')

# visualise_connectivity(S)