import numpy as np
import matplotlib.pyplot as plt
import brian2 as br


# 뉴런 행동 파라미터
taum            = 10 * br.ms        # 뉴런의 막전위 감쇠상수
Ee              = 0 * br.mV         # ??
vt              = -54 * br.mV       # 뉴런의 역치 전위 (Threshold Voltage)
vr              = -60 * br.mV       # 뉴런의 안정 전위 (Resting Potential)
El              = -74 * br.mV       # ??
taue            = 5 * br.ms         # 뉴런 컨덕턴스 감쇠상수

# STDP 학습 파라미터
taupre          = 20 * br.ms        # pre- 강화 연결 감쇠상수
taupost         = taupre            # post- 약화 연결 감쇠상수
gmax            = .01               # 뉴런 컨덕턴스 최대값
dApre           = .01               # pre- 강화 연결 학습 변화율
dApost          = -dApre * taupre / taupost * 1.05 # post- 약화 연결 학습 변화율
dApost          *= gmax
dApre           *= gmax

# 도파민 파라미터
tauc            = 1000 * br.ms      # ??
taud            = 200 * br.ms       # 도파민 감쇠 상수
taus            = 1 * br.ms         # 시냅스 연결강도 
epsilon_dopa    = 5e-3              # ??

# 입력 자극 설정 (Spike)
input_indices   = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]) # 0번 혹은 1번 인덱스의 뉴런에 스파이크 작용
input_times     = np.array([500, 550, 1000, 1010, 1500, 1510, 3500, 3550, 4000, 4010, 4500, 4510]) * br.ms # 작용시각
spike_input     = br.SpikeGeneratorGroup(N=2,
                                         indices=input_indices,
                                         times=input_times,
                                         )

# 뉴런 설정
neurons         = br.NeuronGroup(N=2,
                                 model='''
                                       dv/dt = (ge * (Ee - vr) + (El - v)) / taum  : volt
                                       dge/dt = -ge / taue                         : 1
                                       ''',
                                 threshold='v > vt',
                                 reset='v = vr',
                                 method='exact',
                                 )

neurons.v       = vr
neurons_monitor = br.SpikeMonitor(neurons)

# 시냅스 연결 설정 (pre- 뉴런으로 인한 post- 뉴런의 막전위 증가 정의)
synapse         = br.Synapses(source=spike_input,
                              target=neurons,
                              model=''' s : volt ''',
                              on_pre=''' v += s ''', # Background Spike Generator에서 발생한 Spike로 막전위 증가
                              )

# 다음을 정의하는 시냅스 연결
# i=0 (pre- spike)와 j=0 (pre- neuron)
# i=1 (post- spike)와 j=1 (post- neuron)
synapse.connect(i = [0, 1],
                j = [0, 1])

# pre- 스파이크가 post- 뉴런에 전달하는 막전위 증가량
synapse.s       = 100. * br.mV

# 시냅스 연결 설정 (STDP 학습)
synapse_stdp    = br.Synapses(source=neurons,
                              target=neurons,
                              model='''
                                  mode : 1
                                  dc / dt = -c / tauc : 1               (clock-driven)
                                  dd / dt = -d / taud : 1               (clock-driven)
                                  ds / dt = mode * c * d / taus : 1     (clock-driven)
                                  dApre / dt = -Apre / taupre : 1       (event-driven)
                                  dApost / dt = -Apost / taupost : 1    (event-driven)
                                  ''',
                                  # c, d, s 는 시간에 따라 지수적 감소
                                  # Apre, Apost는 각각 on_pre와 on_post에서 증가
                              on_pre='''
                                  ge += s 
                                  Apre += dApre
                                  c = clip(c + mode * Apost, -gmax, gmax)
                                  s = clip(s + (1-mode) * Apost, -gmax, gmax)
                                  ''',
                                  # pre- 뉴런에서 발생한 Spike로 뉴런 컨덕턴스 증가
                              on_post='''
                                  Apost += dApost
                                  c = clip(c + mode * Apre, -gmax, gmax)
                                  s = clip(s + (1-mode) * Apre, -gmax, gmax)
                                  ''',
                              method='euler',
                              )

# 다음을 정의하는 시냅스 연결
# i=0 (pre- neuron)과 j=1 (post- neuron)
synapse_stdp.connect(i=0, j=1)
synapse_stdp.mode       = 0

# 시냅스 연결강도 (Synapse strength, s(t))
synapse_stdp.s          = 1e-10

# 도파민 작용강도 (Eligibility, c(t))
synapse_stdp.c          = 1e-10

# 외부 도파민 자극강도 (Extracellular dopamine, d(t))
synapse_stdp.d          = 0

synapse_stdp_monitor    = br.StateMonitor(synapse_stdp, ['s', 'c', 'd'], record=[0])

# 입력 자극 설정 (Dopamine)
dopamine_indices        = np.array([0, 0, 0])
dopamine_times          = np.array([3520, 4020, 4520])*br.ms

# 입력 도파민을 스파이크 형태로 전달
dopamine                = br.SpikeGeneratorGroup(N=1,
                                                 indices=dopamine_indices,
                                                 times=dopamine_times,
                                                 )
dopamine_monitor        = br.SpikeMonitor(source=dopamine)

reward                  = br.Synapses(source=dopamine,
                                      target=synapse_stdp,
                                      model=  '''
                                              
                                              ''',
                                      on_pre= '''
                                              d_post += epsilon_dopa
                                              ''',
                                              # ??
                                      method= 'exact',
                                      )

# 다음을 정의하는 시냅스 연결
# 도파민 백그라운드 뉴런 - stdp 학습 시냅스
reward.connect()

# 시뮬레이션 파라미터
simulation_duration     = 6 * br.second

# 기존 STDP
synapse_stdp.mode = 0
br.run(duration=simulation_duration/2)

# 도파민 STDP
synapse_stdp.mode = 1
br.run(duration=simulation_duration/2)

# 출력
dopamine_indices, dopamine_times = dopamine_monitor.it
neurons_indices, neurons_times = neurons_monitor.it
plt.figure(figsize=(12, 6))
plt.subplot(411)
plt.plot([0.05, 2.95], [2.7, 2.7], linewidth=5, color='k')
plt.text(1.5, 3, 'Classical STDP', horizontalalignment='center', fontsize=20)
plt.plot([3.05, 5.95], [2.7, 2.7], linewidth=5, color='k')
plt.text(4.5, 3, 'Dopamine modulated STDP', horizontalalignment='center', fontsize=20)
plt.plot(neurons_times, neurons_indices, 'ob')
plt.plot(dopamine_times, dopamine_indices + 2, 'or')
plt.xlim([0, simulation_duration/br.second])
plt.ylim([-0.5, 4])
plt.yticks([0, 1, 2], ['Pre-neuron', 'Post-neuron', 'Reward'])
plt.xticks([])
plt.subplot(412)
plt.plot(synapse_stdp_monitor.t/br.second, synapse_stdp_monitor.d.T/gmax, 'r-')
plt.xlim([0, simulation_duration/br.second])
plt.ylabel('Extracellular\ndopamine d(t)')
plt.xticks([])
plt.subplot(413)
plt.plot(synapse_stdp_monitor.t/br.second, synapse_stdp_monitor.c.T/gmax, 'b-')
plt.xlim([0, simulation_duration/br.second])
plt.ylabel('Eligibility\ntrace c(t)')
plt.xticks([])
plt.subplot(414)
plt.plot(synapse_stdp_monitor.t/br.second, synapse_stdp_monitor.s.T/gmax, 'g-')
plt.xlim([0, simulation_duration/br.second])
plt.ylabel('Synaptic\nstrength s(t)')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

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

# Synapse Connection : 
    # synapse       = spike_input   <-> neurons
    # synapse_stdp  = neurons       <-> neurons
    # reward        = dopamine      <-> synapse_stdp
visualise_connectivity(synapse)
visualise_connectivity(synapse_stdp)
visualise_connectivity(reward)