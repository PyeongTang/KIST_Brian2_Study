import numpy as np
import matplotlib.pyplot as plt
import brian2 as br

# 뉴런의 행동 특성을 출력해서 볼 수 있도록 스코프를 세워둔다.
br.start_scope()

###############################################################################
############################# 뉴런 파라미터 ####################################
###############################################################################

# 뉴런 행동 파라미터
taum            = 10 * br.ms
Ee              = 0 * br.mV
vt              = -54 * br.mV
vr              = -60 * br.mV
El              = -74 * br.mV
taue            = 5 * br.ms

# STDP 학습 파라미터
taupre          = 20 * br.ms
taupost         = taupre
gmax            = .01
dApre           = .01
dApost          = -dApre * taupre / taupost * 1.05
dApost          *= gmax
dApre           *= gmax

# 도파민 파라미터
tauc            = 1000 * br.ms
taud            = 200 * br.ms
taus            = 1 * br.m
epsilon_dopa    = 5e-3

# 입력 자극 설정 (Spike)
input_indices   = np.array([0, 1, 0, 1 , 1, 0, 0, 1, 0, 1, 1, 0])
input_times     = np.array([500, 550, 1000, 1010, 1500, 1510, 3500, 3550, 4000, 4010, 4500, 4510]) * br.ms
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

# 시냅스 연결 설정
synapse         = br.Synapses(source=spike_input,
                              target=neurons,
                              model=''' s : volt ''',
                              on_pre=''' v += s ''',
                              )

synapse.connect(i = [0, 1],
                j = [0, 1])

synapse.s       = 100. * br.mV

# 학습 알고리즘 설정 (STDP)
synapse_stdp    = br.Synapses(source=neurons,
                              target=neurons,
                              model='''
                                  mode : 1
                                  dc / dt = -c / tauc : 1 (clock-driven)
                                  dd / dt = -d / taud : 1 (clock-driven)
                                  ds / dt = mode * c * d / taus : 1 (clock-driven)
                                  dApre / dt = -Apre / taupre : 1 (event_driven)
                                  dApost / dt = -Apost / taupost : 1 (event_driven)
                                  ''',
                              on_pre='''
                                  ge += s
                                  Apre += dApre
                                  c = clip(c + mode * Apost, -gmax, gmax)
                                  s = clip(s + (1-mode) * Apost, -gmax, gmax)
                                  ''',
                              on_post='''
                                  Apost += dApost
                                  c = clip(c + mode * Apre, -gmax, gmax)
                                  s = clip(s + (1-mode) * Apre, -gmax, gmax)
                                  ''',
                              method='euler',
                              )

synapse_stdp.connect(i=0, j=0)
synapse_stdp.mode       = 0
synapse_stdp.s          = 1e-10
synapse_stdp.c          = 1e-10
synapse_stdp.d          = 0
synapse_stdp_monitor    = br.StateMonitor(synapse_stdp, ['s', 'c', 'd'], record=[0])

# 입력 자극 설정 (Dopamine)
dopamine_indices        = np.array([0, 0, 0])
dopamine_times          = np.array([3520, 4020, 4520])*br.ms
dopamine                = br.SpikeGeneratorGroup(N=1,
                                                 indices=dopamine_indices,
                                                 times=dopamine_times,
                                                 )

reward                  = br.Synapses(source=dopamine,
                                      target=synapse_stdp,
                                      model=  '''
                                              
                                              ''',
                                      on_pre= '''
                                              d_post += eopsilon_dopa
                                              ''',
                                      method= 'exact',
                                      )

reward.connect()

# 시뮬레이션 파라미터
simulation_duration     = 6 * br.second

# 기존 STDP
synapse_stdp.mode = 0
br.run(duration=simulation_duration/2.)

# 도파민 STDP
synapse_stdp.mode = 1
br.run(duration=simulation_duration/2.)
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
plt.xlim([0, simulation_duration/second])
plt.ylim([-0.5, 4])
plt.yticks([0, 1, 2], ['Pre-neuron', 'Post-neuron', 'Reward'])
plt.xticks([])
plt.subplot(412)
plt.plot(synapse_stdp_monitor.t/second, synapse_stdp_monitor.d.T/gmax, 'r-')
plt.xlim([0, simulation_duration/second])
plt.ylabel('Extracellular\ndopamine d(t)')
plt.xticks([])
plt.subplot(413)
plt.plot(synapse_stdp_monitor.t/second, synapse_stdp_monitor.c.T/gmax, 'b-')
plt.xlim([0, simulation_duration/second])
plt.ylabel('Eligibility\ntrace c(t)')
plt.xticks([])
plt.subplot(414)
plt.plot(synapse_stdp_monitor.t/second, synapse_stdp_monitor.s.T/gmax, 'g-')
plt.xlim([0, simulation_duration/second])
plt.ylabel('Synaptic\nstrength s(t)')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()