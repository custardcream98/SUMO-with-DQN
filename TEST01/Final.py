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

HEADERS = ['step']
header_temp = [
    'sortedCarList',
    'queueLength',
    'totalWaitingTime',
    'avgSpeed',
    'avgGapSpace',
]
STRAIGHT = 'straight_'
RIGHT = 'right_'
STRAIGHT_OR_RIGHT = [STRAIGHT, RIGHT]
HAT = '_hat'

for direction in STRAIGHT_OR_RIGHT:
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
        route_build_class.HDV_CAV_RATIO = [0.2, 0.8]
        route_build_class.createRouteFile()

        # 시뮬레이션 max step 및 시나리오(네트워크, 교통량) 파일 명시
        self.sim_max_time = 28800
        self.net_file = "intersection.net.xml"
        self.route_file = "intersection.rou.xml"

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
        self.phaseCount = 4 # 시나리오 파일에서 확인
        self.action_space = gym.spaces.Discrete(self.phaseCount)
        self.lastPhase = self.phaseCount + 1
        self.curPhaseDuration = 1

        # 보상과 MOE
        # 학습 평가를 위한 MOE(measure of effectivness) 초기화
        # 보상: 행동에 따른 상태 개선 정도, 모델은 시뮬레이션 동안 누적 보상을 최대화하도록 학습
        self.moe = 0
        self.throughput = 0
        self.cum_throughput = 0
        traci.close()

        # 예측치 관측치 저장용
        self.result_by_lanes = []

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
        self.cum_throughput = 0
        self.lastPhase = self.phaseCount + 1
        self.curPhaseDuration = 1

        # 시뮬레이션 재시작 및 traci 인터페이스 활성화
        runSUMO = ["sumo-gui", "-n",  self.net_file, '-r', self.route_file, '--random', '--quit-on-end', "--start"]
        traci.start(runSUMO)

        # 상태(초깃값) 리턴
        return state

    def step(self, action): # 시뮬레이션의 매 스텝별 명령 메소드, 상태와 보상 리턴 후 결정된 행동이 입력 
        traci.simulationStep() # 스텝 진행
        # traci.trafficlight.setPhase('gneJ00', action) # 행동에 따른 신호 제어

        state = self.compute_observation(action)
        reward = self.comptue_reward()

        # print(traci.vehicle.getStops(veh_ID))
        # print(traci.vehicle.getEmergencyDecel(veh_ID))
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

    def compute_observation(self, action): # 상태 관측 메소드
        state = []
        state.append(action) # 1. 현재 phase
        
        if self.lastPhase == action:
            self.curPhaseDuration += 1
        else:
            self.curPhaseDuration = 1
            self.lastPhase = action
        state.append(self.curPhaseDuration) # 2. 현재 phase의 지속시간

        # 빨리 지워야해
        waitingTimeArray = [0, 0, 0, 0, 0, 0, 0, 0]
        vehicleCountArray = [0, 0, 0, 0, 0, 0, 0, 0]
        
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
            print()
        
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

                self.incomingLanes[lane_ID][1] = len(sortedCarList) # 차량의 수
                self.incomingLanes[lane_ID][3] = queueLength # 대기행렬의 길이
                self.incomingLanes[lane_ID][5] = totalWaitingTime # 총 대기시간
                self.incomingLanes[lane_ID][7] = avgSpeed_hat # 평균 속력
                self.incomingLanes[lane_ID][9] = avgGapSpace_hat # 평균 차간 간격
        
        # 관측값과 추정값을 저장하는 코드 부탁쓰
        # 가로축
        # traci.simulation.getTime()

        result_currentstep = {}

        ## 관측값 저장 전 init
        for header in HEADERS:
            if header == 'step':
                result_currentstep[header] = traci.simulation.getTime()
            else:
                result_currentstep[header] = 0

        ## 저장
        for lane_ID in self.incomingLanes:
            for i in range(0, 5):
                header = (STRAIGHT if lane_ID.split('_')[1] == '0' else RIGHT) + header_temp[i]
                result_currentstep[header] += self.incomingLanes[lane_ID][2*i]
                result_currentstep[header+HAT] += self.incomingLanes[lane_ID][2*i + 1]

        self.result_by_lanes.append(result_currentstep)


        state = state + waitingTimeArray + vehicleCountArray
        return np.array(state)
    
    def compute_vehicle_state(self, veh_ID):
        lanePosition = traci.vehicle.getLanePosition(veh_ID) # 위치
        waitingTime = traci.vehicle.getWaitingTime(veh_ID) # 대기시간
        speed = traci.vehicle.getSpeed(veh_ID) # 속력
        return (lanePosition, waitingTime, speed)

    def comptue_reward(self):       
        # MOE: 총 대기시간
        sum_waiting_time = 0
        for veh_ID in traci.vehicle.getIDList():
            sum_waiting_time += traci.vehicle.getWaitingTime(veh_ID)
        self.moe += -sum_waiting_time
        return -sum_waiting_time