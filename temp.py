import matplotlib.pyplot as plt
import brian2 as br

# 뉴런의 행동 특성을 출력해서 볼 수 있도록 스코프를 세워둔다.
br.start_scope()

# 뉴런의 수 N 선언
N = 10

# 감쇠 상수 Tau 선언, 단위를 호출할 때엔 메서드 호출이 안정적이다.
tau = 10*br.ms

v0_max = 3.

duration = 1000*br.ms

sigma = .2

# 미분 방정식 선언, 삼 따옴표 내부에 기술하고, 단위를 꼭 남겨준다.
# 분자 / 분모의 단위 또한 중요하다. 여기선 (volt / time)
eqs =   '''
        dv/dt = (v0-v) / tau    : 1
        '''
        
# 뉴런 그룹 클래스로 뉴런 객체를 만들어준다.
G = br.NeuronGroup(N, eqs, threshold='v>1', reset='v=0', method='exact')

# 뉴런의 시작 전압을 (0, 1)사이에서 결정한다.
G.v = 'rand()'

# 뉴런의 시작 전압을 (0, v0_max) 사이에서 인덱스 마다 부여한다.
# G.v0 = 'i * v0_max / (N-1)'

# 뉴런의 상태를 실시간으로 저장하는 모니터를 만들어준다.
stateMon = br.StateMonitor(G, 'v', record=0)

# 스파이크 발생을 저장하는 모니터를 만들어준다.
spikeMon = br.SpikeMonitor(G)

br.run(duration/10.)

plt.plot(stateMon.t/br.ms, stateMon.v[0])