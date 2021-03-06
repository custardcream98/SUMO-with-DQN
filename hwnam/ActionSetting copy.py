import os, sys, gym, gym.spaces, numpy as np

# SUMO 환경변수 경로 설정
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# traci - SUMO 신호 제어 python 인터페이스     
import traci

class SumoEnvironment(gym.Env):
    def __init__(self, env_config):
        # 시뮬레이션 max step 및 시나리오(네트워크, 교통량) 파일 명시
        self.sim_max_time = 3600
        self.net_file = "simple/intersection.net.xml"
        self.route_file = "simple/intersection.rou.xml"

        # SUMO 실행 - 상태 및 행동에 대한 초기화 목적  
        runSUMO = ["sumo", "-n",  self.net_file, '-r', self.route_file, '--random', '--quit-on-end', "--start"]
        traci.start(runSUMO)
        
        # 상태: 모델로의 입력값 (실시간 교통상황 묘사)
        # 행동: 모델의 출력값 (신호 제어, 현 상황에서 가장 적절한 신호(phase)는?)
        # 상태와 행동 벡터에 대한 크기 정의
        
        ## 상태
        ### 각 신호 제어기와 연결된 진입/진출 차선 데이터 초기화
        self.ts_IDs = traci.trafficlight.getIDList() # 신호 제어기 id 추출
        self.incomingLanes = dict() # 진입 차선 딕셔너리 => key:차선id / values 차량의 수, 대기시간  
        self.outgoingLanes = dict() # 진출 차선 딕셔너리 => key:차선id / values 차량들의 id

        for ts in self.ts_IDs: # 신호 제어기별 연결 차선 추출
            links = traci.trafficlight.getControlledLinks(ts) 
            for link in links:
                if link[0][0] not in self.incomingLanes: # 진입 차선 초기화
                    self.incomingLanes[link[0][0]] = []
                if link[0][1] not in self.outgoingLanes: # 진출 차선 초기화
                    self.outgoingLanes[link[0][1]] = []
        
        ### 상태 벡터 크기 정의
        ### 현재 phase id + 현재 phase duration + 진입 차선별 차량의 수 + 진입 차선별 평균 대기시간
        ### 1 + 1 + len(진입 차선 딕셔너리) + len(진입 차선 딕셔너리), 모두 양수
        self.observLength = 1 + 1 + (2 * len(self.incomingLanes))  
        self.observation_space = gym.spaces.Box(low=0, high=sys.float_info.max, shape=(self.observLength,)) # 최솟값, 최댓값, 벡터 크기

        ## 행동
        ### 주어진 상황에 가장 적절한 phase 선택
        ### 행동 벡터의 크기: 시나리오의 phase 개수(Discrete, ex) 0, 1, 2, 3, ...)
        self.phaseCount = 8 # 시나리오 파일에서 확인
        self.action_space = gym.spaces.Discrete(2)
        self.lastPhase = self.phaseCount + 1
        self.curPhaseDuration = 1
        self.curPhase = 0

        # 보상과 MOE
        # 학습 평가를 위한 MOE(measure of effectivness) 초기화
        # 보상: 행동에 따른 상태 개선 정도, 모델은 시뮬레이션 동안 누적 보상을 최대화하도록 학습
        self.moe = 0
        self.throughput = 0
        traci.close()

    def reset(self): # 에피소드 초기화 메소드
        # traci 인터페이스 종료
        if traci.isLoaded():
            traci.close()

        # 변수(상태, 차선 딕셔너리, MOE, 누적 보상) 초기화
        state = np.zeros(self.observLength)
        for lane in self.incomingLanes:
            self.incomingLanes[lane] = []
        for lane in self.outgoingLanes:
            self.outgoingLanes[lane] = []
        self.moe = 0
        self.throughput = 0
        self.lastPhase = self.phaseCount + 1
        self.curPhaseDuration = 1

        # 시뮬레이션 재시작 및 traci 인터페이스 활성화
        runSUMO = ["sumo-gui", "-n",  self.net_file, '-r', self.route_file, '--random', '--quit-on-end', "--start"]
        traci.start(runSUMO)

        # 상태(초깃값) 리턴
        return state

    def step(self, action): # 시뮬레이션의 매 스텝별 명령 메소드, 상태와 보상 리턴 후 결정된 행동이 입력 
        traci.simulationStep() # 1 스텝 진행, curPhaseDuration = 1이기 때문

        # 현재 현시 정보 획득
        curPhase = traci.trafficlight.getPhase('gneJ00') # 0~7
        curPhaseAll = traci.trafficlight.getAllProgramLogics('gneJ00')[0].phases[curPhase] # 페이즈 상세정보
        self.curPhase = curPhase

        while self.curPhaseDuration > curPhaseAll.maxDur:
             # 다음 페이즈로 변경
            self.curPhase += 1
            if self.curPhase == self.phaseCount:
                self.curPhase = 0

            traci.trafficlight.setPhase('gneJ00', self.curPhase) # 행동에 따른 신호 제어
            traci.simulationStep()
            self.curPhaseDuration = 1

        while self.curPhaseDuration < curPhaseAll.minDur:
            traci.simulationStep() # 1 스텝 진행
            self.curPhaseDuration += 1
        
        if action == 1:
            self.curPhase += 1
            if self.curPhase == self.phaseCount:
                self.curPhase = 0
        
        traci.trafficlight.setPhase('gneJ00', self.curPhase) # 행동에 따른 신호 제어

        state = self.compute_observation(self.curPhase)
        reward = self.comptue_reward()

        if traci.simulation.getTime() < self.sim_max_time:
            done = False
        else:
            done = True
            print(self.moe)        
            traci.close()

        info = {}
        return state, reward, done, info

    def compute_observation(self, phase): # 상태 관측 메소드
        state = []
        state.append(phase) # 1. 현재 phase
        
        if self.lastPhase == phase:
            self.curPhaseDuration += 1
        else:
            self.curPhaseDuration = 1
            self.lastPhase = phase

        state.append(self.curPhaseDuration) # 2. 현재 phase의 지속시간

        # 3. 자율차 기반 진입 차로별 추정 값들(대기행렬의 길이, 총 대기시간, 평균속력, 평균 차간간격)
        ## 민혁이가 짜는 중
        waitingTimeArray = [0, 0, 0, 0, 0, 0, 0, 0]
        vehicleCountArray = [0, 0, 0, 0, 0, 0, 0, 0]
        
        # 초기화
        state = state + waitingTimeArray + vehicleCountArray
        return np.array(state)

    def comptue_reward(self):     
        # MOE: 총 대기시간
        sum_waiting_time = 0
        for veh_ID in traci.vehicle.getIDList():
            sum_waiting_time += traci.vehicle.getWaitingTime(veh_ID)
        self.moe += -sum_waiting_time
        return -sum_waiting_time