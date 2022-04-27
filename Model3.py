import os, sys, gym, gym.spaces, numpy as np, csv

# SUMO 환경변수 경로 설정
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# traci - SUMO 신호 제어 python 인터페이스     
import traci

# 효과평가지표(MOE: Measure of Effectiveness): 총 대기시간, 평균 대기행렬 길이, 평균 통과교통량
MOE_HEADERS = [
    'episode',
    'total_waiting_time',
    'avg_queue_length',
    'avg_speed',
    'cum_reward'
]

# 2022.04.25
# MOE 저장 파일명칭과 Reward 수정 후, 학습 수행!
class AV20_M3(gym.Env):
    def __init__(self, env_config):  
        # 효과평가지표 저장 파일 초기화
        with open('AV20_M3.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=MOE_HEADERS)
            writer.writeheader()

        # 시나리오(네트워크, 교통량) 파일 할당
        self.net_file = "intersection.mix.net.xml"
        self.route_file = "intersection.av20.rou.xml"

        # SUMO 초기화: 멤버변수 초기화 목적 
        runSUMO = ["sumo", "-n",  self.net_file, '-r', self.route_file, '--quit-on-end', "--no-step-log", "--duration-log.disable", "--no-warnings", "--start"]
        traci.start(runSUMO)
        
        # 상태: 모델의 입력값 (실시간 교통상황 묘사 -> 진입차선의 교통흐름)         
        ## self.incomingLanes: 진입차선 딕셔너리, 각 진입차선의 교통흐름(차량의 수, 평균 속력, 대기시간, 대기행렬의 길이 등) 저장 
        self.incomingLanes = dict()
        links = traci.trafficlight.getControlledLinks('gneJ00') 
        for link in links:
            if link[0][0] not in self.incomingLanes: # 진입 차선 초기화
                self.incomingLanes[link[0][0]] = []
        
        ## 상태 벡터 크기 정의
        ### 현재 phase id + 현재 phase duration + (진입 차선별 추정 변수값 -> 자율차가 감지할 수 있는 교통흐름 적용)
        ### 추정 변수(4): 차량 수, 대기행렬 길이, 총 대기시간, 평균 속력
        ### 벡터 길이: 1 + 1 + (진입차선 개수 * 4), 모두 양수
        self.state_length = 1 + 1 + (4 * len(self.incomingLanes))  
        self.observation_space = gym.spaces.Box(low=0, high=sys.float_info.max, shape=(self.state_length,)) # 최솟값, 최댓값, 벡터 크기
        self.state = []

        # 행동: 모델의 출력값 (신호 제어 행동)
        ## Model3
        ### 현재 phase 유지 혹은 다음 phase로의 변경 혹은 다음 phase skip
        ### 벡터 길이: 0 또는 1 또는 2 -> 3
        self.action_space = gym.spaces.Discrete(3)
        
        ## 제약조건
        ### 현시 순서 고정, 최소/최대 녹색시간, 주기 최소/최대
        self.phaseCount = 8
        self.curPhase = 0
        self.curPhaseDuration = 0

        # 보상: 모델의 행동에 따른 상태의 개선 정도, 모델은 누적 보상값을 최대화하도록 학습
        self.reward = 0
        self.cum_reward = 0
        self.reward_count = 0
        self.epsode_count = 0

        # MOE
        ## 총 대기시간, 평균 대기행렬 길이, 평균 속력 
        self.moe_waitingtime = 0
        self.moe_queueLength = 0
        self.moe_speed = 0
        
        traci.close()
      
    def reset(self): # 에피소드 초기화
        # 시뮬레이션 시작
        if traci.isLoaded():
            traci.close()
        runSUMO = ["sumo", "-n",  self.net_file, '-r', self.route_file, '--quit-on-end', "--no-step-log", "--duration-log.disable", "--no-warnings", "--start"]
        traci.start(runSUMO)

        # 변수 초기화
        self.state = []
        self.reward = 0
        for lane in self.incomingLanes:
            self.incomingLanes[lane] = []
        self.cum_reward = 0
        self.reward_count = 0
        self.moe_waitingtime = 0
        self.moe_queueLength = 0
        self.moe_speed = 0

        # 3주기 + 첫번째 phase만큼(7초) 시뮬레이션 수행
        while traci.simulation.getTime() < 375:
            traci.simulationStep()      
        self.curPhase = 0
        self.curPhaseDuration = 7

        # 상태(초깃값) 리턴
        return self.compute_observation()

    def step(self, action): # 상태 리턴 -> 행동 결정 -> step: 1. 행동(신호 제어) 적용 / 2. 상태 관측 / 3. 보상 부여 / 4. 상태, 보상, 에피소드 중단 여부 리턴
        # 현재 현시 정보 획득
        self.curPhase = traci.trafficlight.getPhase('gneJ00') # 0~7
        curPhaseAll = traci.trafficlight.getAllProgramLogics('gneJ00')[0].phases[self.curPhase] # phase 상세정보

        if action == 0:
            traci.trafficlight.setPhase('gneJ00', self.curPhase)
            traci.simulationStep()
            self.curPhaseDuration += 1
            self.state = self.compute_observation()
            self.reward = self.comptue_reward()
            
            if self.curPhaseDuration == curPhaseAll.maxDur: # 최대 녹색시간 조건
                self.goto_next_phase_by_max()

        if action == 1: # 다음 phase로 변경
            self.goto_next_phase()
        if action == 2: # 다음 phase 스킵
            self.skip_next_phase()
            
        self.cum_reward += self.reward # 보상 누적
        
        if traci.simulation.getMinExpectedNumber() > 0: 
            done = False # 학습 지속여부
        else: 
            done = True # 에피소드 종료
            self.epsode_count += 1

            # 에피소드 MOE 저장
            self.moe_queueLength = self.moe_queueLength /  self.reward_count
            self.moe_speed = self.moe_speed / self.reward_count
            result_moe = [{
                "episode": self.epsode_count,
                "total_waiting_time": self.moe_waitingtime,
                "avg_queue_length": self.moe_queueLength,
                "avg_speed": self.moe_speed,
                "cum_reward": self.cum_reward
            },]
            with open('AV20_M3.csv', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=MOE_HEADERS)
                writer.writerows(result_moe)                 
            print(self.moe_waitingtime) 

            traci.close()

        info = {}
        return self.state, self.reward, done, info
    
    def sim_min_green(self): # 최소 녹색시간동안 시뮬레이션 진행
        self.curPhase += 1
        if self.curPhase == self.phaseCount:
            self.curPhase = 0
        self.curPhaseDuration = 0
        traci.trafficlight.setPhase('gneJ00', self.curPhase)
        curPhaseAll = traci.trafficlight.getAllProgramLogics('gneJ00')[0].phases[self.curPhase]
        while self.curPhaseDuration < curPhaseAll.minDur:
            traci.simulationStep()
            self.curPhaseDuration += 1
    
    def goto_next_phase(self):
        self.sim_min_green() # yellow phase
        self.sim_min_green() # next phase
        self.state = self.compute_observation()
        self.reward = self.comptue_reward()    
    
    def skip_next_phase(self):
        self.sim_min_green() # yellow phase
        if self.curPhase < 4: # skip next phase
            self.curPhase += 3
        else:
            self.curPhase -= 5
        self.curPhaseDuration = 0
        traci.trafficlight.setPhase('gneJ00', self.curPhase)
        curPhaseAll = traci.trafficlight.getAllProgramLogics('gneJ00')[0].phases[self.curPhase]
        while self.curPhaseDuration < curPhaseAll.minDur:
            traci.simulationStep()
            self.curPhaseDuration += 1
        self.state = self.compute_observation()
        self.reward = self.comptue_reward()
    
    def goto_next_phase_by_max(self):
        self.sim_min_green() # yellow phase
        self.sim_min_green() # next phase
        self.state = self.compute_observation()

    def compute_observation(self): # 상태 관측
        state = []
        state.append(self.curPhase) # 1. 현재 phase
        state.append(self.curPhaseDuration) # 2. 현재 phase의 지속시간
        
        # 차로별 관측값 초기화
        for lane_ID in self.incomingLanes:
            carCount = 0 # 차량의 수
            carCount_hat = 0
            queueLength = 0 # 대기행렬의 길이
            queueLength_hat = 0
            totalWaitingTime = 0 # 총 대기시간
            totalWaitingTime_hat = 0
            avgSpeed = 0 # 평균 속력
            avgSpeed_hat = 0
            avList = dict()
            detectedCarList = dict()
            carList = []

            # 차로별 데이터 수집 
            carCount = traci.lane.getLastStepVehicleNumber(lane_ID)
            if carCount > 0:
                laneLength = traci.lane.getLength(lane_ID) # 차로의 길이
                speed_sum = 0
                # 모든 차량별 데이터 수집
                for veh_ID in traci.lane.getLastStepVehicleIDs(lane_ID):
                    carList.append(veh_ID)
                    carState = self.compute_vehicle_state(veh_ID)
                    # 위치
                    position = laneLength - carState[0]
                    # 대기시간
                    waitingTime = carState[1]
                    totalWaitingTime += waitingTime
                    # 속력
                    speed = carState[2]
                    if speed <= 0.1 and queueLength <= position: # 대기행렬 길이 할당
                        queueLength = position
                    speed_sum += speed
                    if traci.vehicle.getTypeID(veh_ID) == 'CACC': # 자율차 수집
                        avList[veh_ID] = (position, waitingTime, speed, 'CACC')
                # 평균 속력 및 차간간격 계산
                avgSpeed = speed_sum / carCount
            
            # 차로 딕셔서리 초기화
            self.incomingLanes[lane_ID] = [carCount, carCount_hat, queueLength, queueLength_hat, 
            totalWaitingTime, totalWaitingTime_hat, avgSpeed, avgSpeed_hat, avList, detectedCarList, carList]
        
        # 자율차로 감지한 일반차량 추가
        for lane_ID in self.incomingLanes:
            laneLength = traci.lane.getLength(lane_ID)
            for av in self.incomingLanes[lane_ID][8]:
                self.incomingLanes[lane_ID][9][av] = self.incomingLanes[lane_ID][8][av] 
                # leader
                if traci.vehicle.getLeader(av, 30) is not None:
                    leader = traci.vehicle.getLeader(av, 30)[0]
                    if traci.vehicle.getTypeID(leader) != 'CACC' and lane_ID == traci.vehicle.getLaneID(leader):
                        if leader not in self.incomingLanes[lane_ID][9]:
                            carState = self.compute_vehicle_state(leader)
                            position = laneLength - carState[0]
                            waitingTime = carState[1]
                            speed = carState[2]
                            self.incomingLanes[lane_ID][9][leader] = (position, waitingTime, speed, 'hv')
                # follower
                if traci.vehicle.getFollower(av, 30)[1] != -1.0:
                    follower =  traci.vehicle.getFollower(av, 30)[0]
                    if traci.vehicle.getTypeID(follower) != 'CACC' and lane_ID == traci.vehicle.getLaneID(follower):
                        if follower not in self.incomingLanes[lane_ID][9]:
                            carState = self.compute_vehicle_state(follower)
                            position = laneLength - carState[0]
                            waitingTime = carState[1]
                            speed = carState[2]
                            self.incomingLanes[lane_ID][9][follower] = (position, waitingTime, speed, 'hv')
                # left leader / follower
                if len(traci.vehicle.getNeighbors(av, 0^0)) > 0:
                    left_follower = traci.vehicle.getNeighbors(av, 0^0)[0][0]
                    if traci.vehicle.getTypeID(left_follower) != 'CACC':
                        target_lane = traci.vehicle.getLaneID(left_follower)
                        if target_lane in self.incomingLanes and left_follower not in self.incomingLanes[target_lane][9]:
                            carState = self.compute_vehicle_state(left_follower)
                            position = traci.lane.getLength(target_lane) - carState[0]
                            waitingTime = carState[1]
                            speed = carState[2]
                            self.incomingLanes[target_lane][9][left_follower] = (position, waitingTime, speed, 'hv')
                if len(traci.vehicle.getNeighbors(av, 2^0)) > 0:
                    left_leader = traci.vehicle.getNeighbors(av, 2^0)[0][0]
                    if traci.vehicle.getTypeID(left_leader) != 'CACC':
                        target_lane = traci.vehicle.getLaneID(left_leader)
                        if target_lane in self.incomingLanes and left_leader not in self.incomingLanes[target_lane][9]:
                            carState = self.compute_vehicle_state(left_leader)
                            position = traci.lane.getLength(target_lane) - carState[0]
                            waitingTime = carState[1]
                            speed = carState[2]
                            self.incomingLanes[target_lane][9][left_leader] = (position, waitingTime, speed, 'hv')           
                # right leader / follower
                if len(traci.vehicle.getNeighbors(av, 0^1)) > 0:
                    right_follower = traci.vehicle.getNeighbors(av, 0^1)[0][0]
                    if traci.vehicle.getTypeID(right_follower) != 'CACC':
                        target_lane = traci.vehicle.getLaneID(right_follower)
                        if target_lane in self.incomingLanes and right_follower not in self.incomingLanes[target_lane][9]:
                            carState = self.compute_vehicle_state(right_follower)
                            position = traci.lane.getLength(target_lane) - carState[0]
                            waitingTime = carState[1]
                            speed = carState[2]
                            self.incomingLanes[target_lane][9][right_follower] = (position, waitingTime, speed, 'hv')
                if len(traci.vehicle.getNeighbors(av, 2^1)) > 0:
                    right_leader = traci.vehicle.getNeighbors(av, 2^1)[0][0]
                    if traci.vehicle.getTypeID(right_leader) != 'CACC':
                        target_lane = traci.vehicle.getLaneID(right_leader)
                        if target_lane in self.incomingLanes and right_leader not in self.incomingLanes[target_lane][9]:
                            carState = self.compute_vehicle_state(right_leader)
                            position = traci.lane.getLength(target_lane) - carState[0]
                            waitingTime = carState[1]
                            speed = carState[2]
                            self.incomingLanes[target_lane][9][right_leader] = (position, waitingTime, speed, 'hv')
        
        # 차로별 추정값 계산
        for lane_ID in self.incomingLanes:
            if len(self.incomingLanes[lane_ID][9]) > 0:
                sortedCarList = sorted(self.incomingLanes[lane_ID][9].items(), key = lambda item: item[1][0])
                speed_sum = 0
                queueLength = 0
                stoppedCarList = []
                addingWaitingTime = []
                for vehicle in sortedCarList:
                    speed_sum += vehicle[1][2]
                    if vehicle[1][2] <= 0.1:
                        if queueLength <= vehicle[1][0]:
                            queueLength = vehicle[1][0]
                        stoppedCarList.append((vehicle[1][0], vehicle[1][1]))
                        addingWaitingTime.append(vehicle[1][1])
                avgSpeed_hat = speed_sum / len(sortedCarList)

                if len(stoppedCarList) > 1:
                    for i in range(0, len(stoppedCarList) - 1):
                        estimatedCarCount = 0
                        gap = stoppedCarList[i+1][0] - stoppedCarList[i][0]
                        refWaiting = stoppedCarList[i+1][1] + stoppedCarList[i][1]
                        if gap >= 14:
                            estimatedCarCount += 1
                            hidden = int((gap - 14) / 7)
                            estimatedCarCount += hidden
                            for j in range(1, estimatedCarCount + 1):
                                estimatedWaitingTIme = refWaiting * j / (estimatedCarCount + 1)
                                addingWaitingTime.append(estimatedWaitingTIme)
                totalWaitingTime = np.sum(addingWaitingTime)

                self.incomingLanes[lane_ID][1] = len(sortedCarList)
                self.incomingLanes[lane_ID][3] = queueLength
                self.incomingLanes[lane_ID][5] = totalWaitingTime
                self.incomingLanes[lane_ID][7] = avgSpeed_hat
        
        state_carCount = []
        state_queueLength = []
        state_waitingTime = []
        state_avgSpeed = []

        # 차로별 상태 리턴
        for lane_ID in self.incomingLanes:
            state_carCount.append(self.incomingLanes[lane_ID][1])
            state_queueLength.append(self.incomingLanes[lane_ID][3])
            state_waitingTime.append(self.incomingLanes[lane_ID][5])
            state_avgSpeed.append(self.incomingLanes[lane_ID][7])

        state = state + state_carCount + state_queueLength + state_waitingTime + state_avgSpeed

        return np.array(state)
    
    def compute_vehicle_state(self, veh_ID):
        lanePosition = traci.vehicle.getLanePosition(veh_ID) # 위치
        waitingTime = traci.vehicle.getWaitingTime(veh_ID) # 대기시간
        speed = traci.vehicle.getSpeed(veh_ID) # 속력
        return (lanePosition, waitingTime, speed)
    
    def comptue_reward(self):
        waitingTime_hat = 0
        self.reward_count += 1
    
        waitingTime = 0
        queueLength = 0
        weightedSpeed = 0
        weights = []

        for lane_ID in self.incomingLanes:
            # MOE
            waitingTime += self.incomingLanes[lane_ID][4]
            queueLength += self.incomingLanes[lane_ID][2]
            weightedSpeed += self.incomingLanes[lane_ID][6]
            weights.append(self.incomingLanes[lane_ID][0])

            # reward(추정값)
            waitingTime_hat += self.incomingLanes[lane_ID][5]

        if np.sum(weights) == 0:
            weightedSpeed = 0
        else:
            weightedSpeed = weightedSpeed / np.sum(weights) 
        self.moe_waitingtime += waitingTime
        self.moe_queueLength += queueLength
        self.moe_speed += weightedSpeed

        return -(waitingTime_hat)
