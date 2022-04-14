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

# HEADERS = ['episode_id', 'step']
HEADERS = ['step']

header_temp = [
    'sortedCarList',
    'queueLength',
    'totalWaitingTime',
    'avgSpeed',
    'avgGapSpace',
]
STRAIGHT = 'straight_'
LEFT = 'left_'
STRAIGHT_OR_LEFT = [STRAIGHT, LEFT]
HAT = '_hat'

for direction in STRAIGHT_OR_LEFT:
    for header in header_temp:
        header = direction + header
        HEADERS.append(header)
        HEADERS.append(header + HAT)

class SumoEnvironment(gym.Env):
    def __init__(self, env_config):
        # result 헤더 입력
        with open('result.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=HEADERS)
            writer.writeheader()

        # 일반차량, 자율주행차량 비율 결정하여 intersection.rou.xml 생성
        route_build_class = route_builder.RouteFile()
        route_build_class.HDV_CAV_RATIO = [0.6, 0.4]
        route_build_class.createRouteFile()

        # 시뮬레이션 max step 및 시나리오(네트워크, 교통량) 파일 명시
        self.sim_max_time = 28800
        self.net_file = "intersection.net.xml"
        self.route_file = "intersection.rou.xml"

        # SUMO 실행 - 상태 및 행동에 대한 초기화 목적  
        runSUMO = ["sumo", "-n",  self.net_file, '-r', self.route_file, '--random', '--quit-on-end', "--start"]
        traci.start(runSUMO)
        
        # 22.04.14
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
        ### 차량 수 + 대기행렬 길이 + 총 대기시간 + 평균 속력 + 평균 차간 간격
        ### 1 + 1 + len(진입 차선 딕셔너리) * 5, 모두 양수
        self.observLength = 1 + 1 + (5 * len(self.incomingLanes))  
        self.observation_space = gym.spaces.Box(low=0, high=sys.float_info.max, shape=(self.observLength,)) # 최솟값, 최댓값, 벡터 크기

        ## 행동
        ### Model Type 1
        ### 주기 고정, 현시 순서 고정, 최소/최대 녹색시간 부여
        ### 행동: 현재 phase 유지 혹은 다음 phase로의 변경
        ### 벡터 크기: 0 또는 1 -> 2
        self.phaseCount = 8 # 시나리오 파일에서 확인
        self.action_space = gym.spaces.Discrete(2)
        self.curPhaseDuration = 0
        self.curPhase = 0
        self.cycleLength = 122
        self.cycleElapsedTime = 0
        self.remainPhasesMinDuration = 0

        # 보상과 MOE
        # 학습 평가를 위한 MOE(measure of effectivness) 초기화
        # 보상: 행동에 따른 상태 개선 정도, 모델은 시뮬레이션 동안 누적 보상을 최대화하도록 학습
        self.moe = 0
        traci.close()

        # 예측치 관측치 저장용
        self.result_by_lanes = []

    def reset(self): # 에피소드 초기화 메소드
        # traci 인터페이스 종료
        if traci.isLoaded():
            traci.close()

        # 변수 초기화
        state = np.zeros(self.observLength)
        for lane in self.incomingLanes:
            self.incomingLanes[lane] = []
        
        self.moe = 0
        self.curPhase = 0
        self.curPhaseDuration = 0
        self.cycleElapsedTime = 0
        self.remainPhasesMinDuration = 0

        # 시뮬레이션 재시작
        runSUMO = ["sumo-gui", "-n",  self.net_file, '-r', self.route_file, '--random', '--quit-on-end', "--start"]
        traci.start(runSUMO)
        traci.simulationStep() # 시뮬레이션 시작 (한번 스텝을 진행시켜야 0초)

        self.remainPhasesMinDuration = self.compute_remainPhasesMinDur(self.curPhase + 1)

        # 상태(초깃값) 리턴
        return state

    def step(self, action): # 시뮬레이션의 매 스텝별 명령 메소드, 상태와 보상 리턴 후 결정된 행동이 입력 
       # 현재 현시 정보 획득
        self.curPhase = traci.trafficlight.getPhase('gneJ00') # 0~7
        curPhaseAll = traci.trafficlight.getAllProgramLogics('gneJ00')[0].phases[self.curPhase] # 페이즈 상세정보

        if self.curPhaseDuration < curPhaseAll.minDur:
            self.sim_min_green(curPhaseAll.minDur) # 최소 녹색시간만큼 진행

        if action == 1:
            self.goto_next_phase()
        else:
            traci.trafficlight.setPhase('gneJ00', self.curPhase)
            traci.simulationStep()
            self.curPhaseDuration += 1
            self.cycleElapsedTime += 1
            
            if self.curPhaseDuration == curPhaseAll.maxDur:
                self.goto_next_phase() # 최대 녹색시간을 만족했으면, 다음 페이즈로 변경
            
            if (self.cycleLength - self.cycleElapsedTime) == self.remainPhasesMinDuration:
                self.goto_next_phase() # 주기 고정 - 남은 phase의 최소 녹색시간만큼 진행

        state = self.compute_observation()
        reward = self.comptue_reward()

        if traci.simulation.getTime() < self.sim_max_time:
            done = False
        else:
            done = True
            print(self.moe)        
            traci.close()

            with open('result.csv', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=HEADERS)
                writer.writerows(self.result_by_lanes)

        info = {}
        return state, reward, done, info
    
    def compute_remainPhasesMinDur(self, curPhase):
        remainMinDur = 0
        if curPhase == self.phaseCount:
            curPhase = 0
        for i in range(curPhase, self.phaseCount):
            remainMinDur += traci.trafficlight.getAllProgramLogics('gneJ00')[0].phases[i].minDur
        return remainMinDur

    def sim_min_green(self, min):
        while self.curPhaseDuration < min:
            traci.simulationStep()
            self.curPhaseDuration += 1
            self.cycleElapsedTime += 1
    
    def goto_next_phase(self):
        self.curPhase += 1
        if self.curPhase == self.phaseCount:
            self.curPhase = 0
            self.cycleElapsedTime = 0
        self.remainPhasesMinDuration = self.compute_remainPhasesMinDur(self.curPhase + 1)
        traci.trafficlight.setPhase('gneJ00', self.curPhase)
        self.curPhaseDuration = 0
        curPhaseAll = traci.trafficlight.getAllProgramLogics('gneJ00')[0].phases[self.curPhase]
        self.sim_min_green(curPhaseAll.minDur)

        if self.curPhase != 6:
            if self.curPhaseDuration == curPhaseAll.maxDur: 
                self.goto_next_phase()
            elif (self.cycleLength - self.cycleElapsedTime) == self.remainPhasesMinDuration:
                self.goto_next_phase()
        else: # 마지막 phase는 잔여 시간 할당
            while (self.cycleLength - self.cycleElapsedTime) > self.remainPhasesMinDuration:
                traci.trafficlight.setPhase('gneJ00', self.curPhase)
                traci.simulationStep()
                self.curPhaseDuration += 1
                self.cycleElapsedTime += 1
            self.goto_next_phase()

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
            avgGapSpace = 0 # 평균 차간간격
            avgGapSpace_hat = 0
            avList = dict()
            detectedCarList = dict()
            carList = []

            # 차로별 데이터 수집 
            carCount = traci.lane.getLastStepVehicleNumber(lane_ID)
            if carCount > 0:
                laneLength = traci.lane.getLength(lane_ID) # 차로의 길이
                positions = []
                speed_sum = 0
                # 모든 차량별 데이터 수집
                for veh_ID in traci.lane.getLastStepVehicleIDs(lane_ID):
                    carList.append(veh_ID)
                    carState = self.compute_vehicle_state(veh_ID)
                    # 위치
                    position = laneLength - carState[0]
                    positions.append(position)
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
                avgGapSpace = np.std(positions)
            
            # 차로 딕셔서리 초기화
            self.incomingLanes[lane_ID] = [carCount, carCount_hat, queueLength, queueLength_hat, 
            totalWaitingTime, totalWaitingTime_hat, avgSpeed, avgSpeed_hat, avgGapSpace, avgGapSpace_hat, avList, detectedCarList, carList]
        
        # 자율차로 감지한 일반차량 추가
        for lane_ID in self.incomingLanes:
            laneLength = traci.lane.getLength(lane_ID)
            for av in self.incomingLanes[lane_ID][10]:
                self.incomingLanes[lane_ID][11][av] = self.incomingLanes[lane_ID][10][av] 
                # leader
                if traci.vehicle.getLeader(av, 30) is not None:
                    leader = traci.vehicle.getLeader(av, 30)[0]
                    if traci.vehicle.getTypeID(leader) != 'CACC' and lane_ID == traci.vehicle.getLaneID(leader):
                        if leader not in self.incomingLanes[lane_ID][11]:
                            carState = self.compute_vehicle_state(leader)
                            position = laneLength - carState[0]
                            waitingTime = carState[1]
                            speed = carState[2]
                            self.incomingLanes[lane_ID][11][leader] = (position, waitingTime, speed, 'hv')
                # follower
                if traci.vehicle.getFollower(av, 30)[1] != -1.0:
                    follower =  traci.vehicle.getFollower(av, 30)[0]
                    if traci.vehicle.getTypeID(follower) != 'CACC' and lane_ID == traci.vehicle.getLaneID(follower):
                        if follower not in self.incomingLanes[lane_ID][11]:
                            carState = self.compute_vehicle_state(follower)
                            position = laneLength - carState[0]
                            waitingTime = carState[1]
                            speed = carState[2]
                            self.incomingLanes[lane_ID][11][follower] = (position, waitingTime, speed, 'hv')
                # left leader / follower
                if len(traci.vehicle.getNeighbors(av, 0^0)) > 0:
                    left_follower = traci.vehicle.getNeighbors(av, 0^0)[0][0]
                    if traci.vehicle.getTypeID(left_follower) != 'CACC':
                        target_lane = traci.vehicle.getLaneID(left_follower)
                        if target_lane in self.incomingLanes and left_follower not in self.incomingLanes[target_lane][11]:
                            carState = self.compute_vehicle_state(left_follower)
                            position = traci.lane.getLength(target_lane) - carState[0]
                            waitingTime = carState[1]
                            speed = carState[2]
                            self.incomingLanes[target_lane][11][left_follower] = (position, waitingTime, speed, 'hv')
                if len(traci.vehicle.getNeighbors(av, 2^0)) > 0:
                    left_leader = traci.vehicle.getNeighbors(av, 2^0)[0][0]
                    if traci.vehicle.getTypeID(left_leader) != 'CACC':
                        target_lane = traci.vehicle.getLaneID(left_leader)
                        if target_lane in self.incomingLanes and left_leader not in self.incomingLanes[target_lane][11]:
                            carState = self.compute_vehicle_state(left_leader)
                            position = traci.lane.getLength(target_lane) - carState[0]
                            waitingTime = carState[1]
                            speed = carState[2]
                            self.incomingLanes[target_lane][11][left_leader] = (position, waitingTime, speed, 'hv')           
                # right leader / follower
                if len(traci.vehicle.getNeighbors(av, 0^1)) > 0:
                    right_follower = traci.vehicle.getNeighbors(av, 0^1)[0][0]
                    if traci.vehicle.getTypeID(right_follower) != 'CACC':
                        target_lane = traci.vehicle.getLaneID(right_follower)
                        if target_lane in self.incomingLanes and right_follower not in self.incomingLanes[target_lane][11]:
                            carState = self.compute_vehicle_state(right_follower)
                            position = traci.lane.getLength(target_lane) - carState[0]
                            waitingTime = carState[1]
                            speed = carState[2]
                            self.incomingLanes[target_lane][11][right_follower] = (position, waitingTime, speed, 'hv')
                if len(traci.vehicle.getNeighbors(av, 2^1)) > 0:
                    right_leader = traci.vehicle.getNeighbors(av, 2^1)[0][0]
                    if traci.vehicle.getTypeID(right_leader) != 'CACC':
                        target_lane = traci.vehicle.getLaneID(right_leader)
                        if target_lane in self.incomingLanes and right_leader not in self.incomingLanes[target_lane][11]:
                            carState = self.compute_vehicle_state(right_leader)
                            position = traci.lane.getLength(target_lane) - carState[0]
                            waitingTime = carState[1]
                            speed = carState[2]
                            self.incomingLanes[target_lane][11][right_leader] = (position, waitingTime, speed, 'hv')
        
        # 차로별 추정값 계산
        for lane_ID in self.incomingLanes:
            if len(self.incomingLanes[lane_ID][11]) > 0:
                sortedCarList = sorted(self.incomingLanes[lane_ID][11].items(), key = lambda item: item[1][0])
                positions = []
                speed_sum = 0
                queueLength = 0
                stoppedCarList = []
                addingWaitingTime = []
                for vehicle in sortedCarList:
                    positions.append(vehicle[1][0])
                    speed_sum += vehicle[1][2]
                    if vehicle[1][2] <= 0.1:
                        if queueLength <= vehicle[1][0]:
                            queueLength = vehicle[1][0]
                        stoppedCarList.append((vehicle[1][0], vehicle[1][1]))
                        addingWaitingTime.append(vehicle[1][1])
                avgSpeed_hat = speed_sum / len(sortedCarList)
                avgGapSpace_hat = np.std(positions)

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
                self.incomingLanes[lane_ID][9] = avgGapSpace_hat
        
        state_carCount = []
        state_queueLength = []
        state_waitingTime = []
        state_avgSpeed = []
        state_avgGapSpace = []

        # 차로별 상태 리턴
        for lane_ID in self.incomingLanes:
            state_carCount.append(self.incomingLanes[lane_ID][1])
            state_queueLength.append(self.incomingLanes[lane_ID][3])
            state_waitingTime.append(self.incomingLanes[lane_ID][5])
            state_avgSpeed.append(self.incomingLanes[lane_ID][7])
            state_avgGapSpace.append(self.incomingLanes[lane_ID][9])

        state = state + state_carCount + state_queueLength + state_waitingTime + state_avgSpeed + state_avgGapSpace

        result_currentstep = {}

        ## 관측값 저장 전 init
        for header in HEADERS:
            # if header == 'episode_id':
            #     result_currentstep[header] = self.epi
            if header == 'step':
                result_currentstep[header] = traci.simulation.getTime()
            else:
                result_currentstep[header] = 0

        ## 저장
        for lane_ID in self.incomingLanes:
            for i in range(0, 5):
                header = (STRAIGHT if lane_ID.split('_')[1] == '0' else LEFT) + header_temp[i]
                result_currentstep[header] += self.incomingLanes[lane_ID][2*i]
                result_currentstep[header+HAT] += self.incomingLanes[lane_ID][2*i + 1]

        self.result_by_lanes.append(result_currentstep)

        return np.array(state)
    
    def compute_vehicle_state(self, veh_ID):
        lanePosition = traci.vehicle.getLanePosition(veh_ID) # 위치
        waitingTime = traci.vehicle.getWaitingTime(veh_ID) # 대기시간
        speed = traci.vehicle.getSpeed(veh_ID) # 속력
        return (lanePosition, waitingTime, speed)
    
    def comptue_reward(self):
        # MOE: 총 대기시간
        waitingTime = 0
        waitingTime_hat = 0
        for lane_ID in self.incomingLanes:
            waitingTime += self.incomingLanes[lane_ID][4]
            waitingTime_hat += self.incomingLanes[lane_ID][5]
        
        self.moe += -waitingTime
        return -waitingTime_hat