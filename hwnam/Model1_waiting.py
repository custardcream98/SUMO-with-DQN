import os, sys, gym, gym.spaces, numpy as np, csv
import route_builder

# SUMO 환경변수 경로 설정
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# traci - SUMO 신호 제어 python 인터페이스     
import traci

MOE_HEADERS = [
    'episode',
    'moe_totalWaitingTime',
    'moe_queueLength',
    'moe_avgSpeed',
    'simulation_step',
    'cum_reward'
]
# HEADERS = ['episode_id', 'step']
# HEADERS = ['step']

# header_temp = [
#     'sortedCarList',
#     'queueLength',
#     'totalWaitingTime',
#     'avgSpeed',
#     'avgGapSpace',
# ]
# STRAIGHT = 'straight_'
# LEFT = 'left_'
# STRAIGHT_OR_LEFT = [STRAIGHT, LEFT]
# HAT = '_hat'

# for direction in STRAIGHT_OR_LEFT:
#     for header in header_temp:
#         header = direction + header
#         HEADERS.append(header)
#         HEADERS.append(header + HAT)

class SumoEnvironment(gym.Env):
    def __init__(self, env_config):
        # result 헤더 입력
        # with open('result.csv', 'w', newline='', encoding='utf-8') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=HEADERS)
        #     writer.writeheader()
        with open('result_model1.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=MOE_HEADERS)
            writer.writeheader()

        # 일반차량, 자율주행차량 비율 결정하여 intersection.rou.xml 생성
        # route_build_class = route_builder.RouteFile()
        # route_build_class.HDV_CAV_RATIO = [0.2, 0.8]
        # route_build_class.createRouteFile()

        # 시나리오(네트워크, 교통량) 파일 명시
        self.net_file = "intersection.net.xml"
        self.route_file = "intersection_case5.rou.xml"

        # SUMO 실행 - 상태 및 행동에 대한 초기화 목적  
        runSUMO = ["sumo", "-n",  self.net_file, '-r', self.route_file, '--random', '--quit-on-end', "--no-step-log", "--duration-log.disable", "--no-warnings", "--start"]
        traci.start(runSUMO)
        
        # 22.04.20
        # 상태: 모델로의 입력값 (실시간 교통상황 묘사)
        # 행동: 모델의 출력값 (신호 제어)
        
        # 상태와 행동 벡터에 대한 크기 정의
        ## 상태
        ### 각 신호 제어기와 연결된 진입/진출 차선 데이터 초기화
        self.ts_IDs = traci.trafficlight.getIDList() # 신호 제어기 id 추출
        self.incomingLanes = dict() # 진입 차선 딕셔너리 => key:차선id
        
        for ts in self.ts_IDs: # 신호 제어기별 연결 차선 추출
            links = traci.trafficlight.getControlledLinks(ts) 
            for link in links:
                if link[0][0] not in self.incomingLanes: # 진입 차선 초기화
                    self.incomingLanes[link[0][0]] = []
        
        ### 상태 벡터 크기 정의
        ### 현재 phase id + 현재 phase duration + (진입 차선별 추정값들...) 
        ### 차량 수 + 대기행렬 길이 + 총 대기시간 + 평균 속력
        ### 1 + 1 + len(진입 차선 딕셔너리) * 4, 모두 양수
        self.observLength = 1 + 1 + (4 * len(self.incomingLanes))  
        self.observation_space = gym.spaces.Box(low=0, high=sys.float_info.max, shape=(self.observLength,)) # 최솟값, 최댓값, 벡터 크기
        self.state = []

        ## 행동
        ### Model Type 1
        ### 현시 순서 고정, 최소/최대 녹색시간 부여, 주기 고정
        ### 행동: 현재 phase 유지 혹은 다음 phase로의 변경
        ### 벡터 크기: 0 또는 1 -> 2
        self.phaseCount = 8
        self.action_space = gym.spaces.Discrete(2)
        self.curPhaseDuration = 0
        self.curPhase = 0
        self.cycleLength = 122
        self.cycleElapsedTime = 0
        self.remainPhasesMinDuration = 0

        # 보상과 MOE
        # 학습 평가를 위한 MOE(measure of effectivness) 초기화
        # 보상: 행동에 따른 상태 개선 정도, 모델은 시뮬레이션 동안 누적 보상을 최대화하도록 학습
        self.moe_totalWaitingTime = 0
        self.moe_queueLength = 0
        self.moe_avgSpeed = 0
        self.reward = 0
        self.cum_reward = 0

        self.reward_count = 0
        traci.close()

        # 예측치/관측치 저장용
        self.result_by_lanes = []
        self.epsode_count = 0

    def reset(self): # 에피소드 초기화 메소드
        # traci 인터페이스 종료
        if traci.isLoaded():
            traci.close()

        # 변수 초기화
        state = np.zeros(self.observLength)
        for lane in self.incomingLanes:
            self.incomingLanes[lane] = []

        # 시뮬레이션 재시작
        runSUMO = ["sumo", "-n",  self.net_file, '-r', self.route_file, '--random', '--quit-on-end', "--no-step-log", "--duration-log.disable", "--no-warnings", "--start"]
        traci.start(runSUMO)

        self.moe_totalWaitingTime = 0
        self.moe_queueLength = 0
        self.moe_avgSpeed = 0
        self.cum_reward = 0
        self.reward_count = 0

        self.state = []
        self.reward = 0

        # 시뮬레이션 초기화
        while traci.simulation.getTime() < 374:
            traci.simulationStep() 
        
        self.curPhase = 0
        self.curPhaseDuration = 7
        self.cycleElapsedTime = 7
        self.remainPhasesMinDuration = self.compute_remainPhasesMinDur(self.curPhase + 1)

        # 상태(초깃값) 리턴
        return self.compute_observation()

    def step(self, action): # 시뮬레이션의 매 스텝별 명령 메소드, 상태와 보상 리턴 후 결정된 행동이 입력 
       # 현재 현시 정보 획득
        self.curPhase = traci.trafficlight.getPhase('gneJ00') # 0~7
        curPhaseAll = traci.trafficlight.getAllProgramLogics('gneJ00')[0].phases[self.curPhase] # 페이즈 상세정보

        action = 0
        if action == 1:
            self.goto_next_phase_by_action()
        else:
            traci.trafficlight.setPhase('gneJ00', self.curPhase)
            traci.simulationStep()
            self.curPhaseDuration += 1
            self.cycleElapsedTime += 1
            self.state = self.compute_observation()
            self.reward = self.comptue_reward()

            if (self.cycleLength - self.cycleElapsedTime) == self.remainPhasesMinDuration:
                self.goto_next_phase_by_mindur() # 주기 고정, 최소 녹색시간, 잔여 phase 실행
            
            if self.curPhaseDuration == curPhaseAll.maxDur:
                self.goto_next_phase_by_max() # 최대 녹색시간을 만족했으면, 다음 페이즈로 변경
        
        self.cum_reward += self.reward
        if traci.simulation.getMinExpectedNumber() > 0:
            done = False
        else:
            done = True
            print(self.moe_totalWaitingTime)        

            self.epsode_count += 1

            moe_queueLength = self.moe_queueLength / self.reward_count
            moe_avgSpeed = self.moe_avgSpeed / self.reward_count

            result_moe = [{
                "episode": self.epsode_count,
                "moe_totalWaitingTime": self.moe_totalWaitingTime,
                "moe_queueLength": moe_queueLength,
                "moe_avgSpeed": moe_avgSpeed,
                "simulation_step": traci.simulation.getTime(),
                "cum_reward": self.cum_reward
            },]

            traci.close()
            # with open('result.csv', 'a', newline='', encoding='utf-8') as csvfile:
            #     writer = csv.DictWriter(csvfile, fieldnames=HEADERS)
            #     writer.writerows(self.result_by_lanes)
            with open('result_model1.csv', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=MOE_HEADERS)
                writer.writerows(result_moe)

        info = {}
        return self.state, self.reward, done, info
    
    def compute_remainPhasesMinDur(self, curPhase):
        remainMinDur = 0
        if curPhase == self.phaseCount:
            curPhase = 0
        for i in range(curPhase, self.phaseCount):
            remainMinDur += traci.trafficlight.getAllProgramLogics('gneJ00')[0].phases[i].minDur
        return remainMinDur

    def sim_min_green(self):
        self.curPhase += 1
        if self.curPhase == self.phaseCount:
            self.curPhase = 0
            self.cycleElapsedTime = 0
        self.curPhaseDuration = 0
        self.remainPhasesMinDuration = self.compute_remainPhasesMinDur(self.curPhase + 1) 
        traci.trafficlight.setPhase('gneJ00', self.curPhase)
        curPhaseAll = traci.trafficlight.getAllProgramLogics('gneJ00')[0].phases[self.curPhase]
        while self.curPhaseDuration < curPhaseAll.minDur:
            traci.simulationStep()
            self.curPhaseDuration += 1
            self.cycleElapsedTime += 1
    
    def sim_remain_phase(self):
        while (self.cycleLength - self.cycleElapsedTime) > self.remainPhasesMinDuration:
            traci.trafficlight.setPhase('gneJ00', self.curPhase)
            traci.simulationStep()
            self.curPhaseDuration += 1
            self.cycleElapsedTime += 1
        self.sim_min_green() # yellow phase
        self.sim_min_green() # first phase

    def goto_next_phase_by_action(self):
        self.sim_min_green() # yellow phase
        self.sim_min_green() # next phase
        self.state = self.compute_observation()
        self.reward = self.comptue_reward()
        if self.curPhase == 6:
            self.sim_remain_phase()
            self.state = self.compute_observation()

    def goto_next_phase_by_mindur(self):
        start = self.curPhase + 1
        for i in range(start, self.phaseCount):
            self.sim_min_green()
        self.sim_min_green() # first phase
        self.state = self.compute_observation()

    def goto_next_phase_by_max(self):
        self.sim_min_green() # yellow phase
        self.sim_min_green() # next phase
        if self.curPhase == 6:
            self.sim_remain_phase()
            self.state = self.compute_observation()
        else:
            self.state = self.compute_observation()

    def compute_observation(self): # 상태 관측 메소드
        state = []
        state.append(self.curPhase) # 1. 현재 phase
        state.append(self.curPhaseDuration) # 2. 현재 phase의 지속시간
        
        # 차로별 관측값 초기화
        for lane_ID in self.incomingLanes:
            # 초기화
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
        self.reward_count += 1
    
        waitingTime = 0
        queueLength = 0
        avgSpeed = 0
        
        # 보상: 추정 대기시간
        waitingTime_hat = 0

        for lane_ID in self.incomingLanes:
            waitingTime += self.incomingLanes[lane_ID][4]
            queueLength += self.incomingLanes[lane_ID][2]
            avgSpeed += self.incomingLanes[lane_ID][6]
            waitingTime_hat += self.incomingLanes[lane_ID][5]
        avgSpeed = avgSpeed / len(self.incomingLanes)
        
        self.moe_totalWaitingTime += waitingTime
        self.moe_queueLength += queueLength
        self.moe_avgSpeed += avgSpeed

        return -(waitingTime_hat)