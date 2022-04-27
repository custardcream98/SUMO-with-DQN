from xml.etree.ElementTree import Element, ElementTree
import random

class RouteFile():
    HORIZONTAL_STRAIGHT_PROBABILITY = [0.12,0.12,0.096,0.096,0.096,0.072,0.072,0.072,0.12,0.12,0.096,0.096,0.096,0.072,0.072,0.072] # 6 to 4 / 4 to 6
    HORIZONTAL_LEFT_PROBABILITY = [0.024,0.024,0.024,0.024,0.024,0.024,0.024,0.024,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012] # 6 to 5 / 4 to 7
    VERTICAL_STRAIGHT_PROBABILITY = [0.048,0.036,0.06,0.048,0.036,0.06,0.048,0.036,0.048,0.036,0.06,0.048,0.036,0.06,0.048,0.036] # 5 to 7 / 7 to 5
    VERTICAL_LEFT_PROBABILITY = [0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.024,0.024,0.024,0.024,0.024,0.024,0.024,0.024] # 5 to 4 / 7 to 6

    CASE = [
            HORIZONTAL_STRAIGHT_PROBABILITY,
            HORIZONTAL_LEFT_PROBABILITY,
            VERTICAL_STRAIGHT_PROBABILITY,
            VERTICAL_LEFT_PROBABILITY,
        ]
    HDV_CAV_RATIO = [0.5, 0.5]

    DIRECTIONS = [
        ["-gneE6", "gneE4"],  # h-s
        ["-gneE4", "gneE6"],  # h-s
        ["-gneE6", "gneE5"],  # h-l
        ["-gneE4", "gneE7"],  # h-l
        ["-gneE5", "gneE7"],  # v-s
        ["-gneE7", "gneE5"],  # v-s
        ["-gneE5", "gneE4"],  # v-l
        ["-gneE7", "gneE6"],  # v-l
    ]

    def createRouteFile(self):
        # vehicle types
        root = Element('routes')
        element1 = Element('vType')
        element1.set('id', 'Krauss')
        element1.set('type', 'passenger')
        element1.set('sigma', '1')

        element2 = Element('vType')
        element2.set('id', 'CACC')
        element2.set('type', 'passenger')
        element2.set('color', '255,20,20')
        element2.set('carFollowModel', 'CACC')

        root.append(element1)
        root.append(element2)

        case_made = []

        for i in range(0, len(self.HORIZONTAL_STRAIGHT_PROBABILITY)):
            while True:
                current_case = random.randint(0, 15)
                if current_case not in case_made:
                    case_made.append(current_case)
                    break

            for j in range(0, 4):
                # AV 차량만 있을 때는 일반차량 flow는 만들지 않도록 함
                if self.HDV_CAV_RATIO[0] == 0:
                    for k in range(0, 2):
                        element = self.makeflow(i, j, self.DIRECTIONS[2 * j + k][0], self.DIRECTIONS[2 * j + k][1],
                                                self.HDV_CAV_RATIO[1] * self.CASE[j][case_made[i]],
                                                "CACC", k)
                        root.append(element)
                # HV 차량만 있을 때는 자율차량 flow는 만들지 않도록 함
                elif self.HDV_CAV_RATIO[1] == 0:
                    for k in range(0, 2):
                        element = self.makeflow(i, j, self.DIRECTIONS[2 * j + k][0], self.DIRECTIONS[2 * j + k][1],
                                                self.HDV_CAV_RATIO[0] * self.CASE[j][case_made[i]],
                                                "KRAUSS", k)
                        root.append(element)
                # 둘 다 만들어야 할 때는 아래와 같이 만듬
                else:
                    # k는 왼쪽에서 오른쪽 방향이 0이라면, 그 반대 방향은 1로 만들어 생성 규칙을 지키기 위한 값
                    for k in range(0, 2):
                        element_K = self.makeflow(i, j, self.DIRECTIONS[2 * j + k][0], self.DIRECTIONS[2 * j + k][1],
                                                  self.HDV_CAV_RATIO[0] * self.CASE[j][case_made[i]],
                                                  "KRAUSS", k)
                        element_C = self.makeflow(i, j, self.DIRECTIONS[2 * j + k][0], self.DIRECTIONS[2 * j + k][1],
                                                  self.HDV_CAV_RATIO[1] * self.CASE[j][case_made[i]],
                                                  "CACC", k)
                        root.append(element_K)
                        root.append(element_C)

        self._pretty_print(root)
        tree = ElementTree(root)

        fileName = "intersection.rou.xml"
        with open(fileName, "wb") as file:
            tree.write(file, encoding='utf-8', xml_declaration=False)

    # flow 한줄 만드는 코드
    def makeflow(self, x, y, string_from, string_to, probability, v_type, way):
        element = Element('flow')
        begin = x * 1800 + 1 if x != 0 else 0
        end = (x + 1) * 1800

        if v_type == "KRAUSS":
            element.set('id', f'HV{x * 8 + y * 2 + (1 + int(way)):03}')
            element.set('type', 'Krauss')
        elif v_type == "CACC":
            element.set('id', f'AV{x * 8 + y * 2 + (1 + int(way)):03}')
            element.set('type', 'CACC')

        element.attrib['begin'] = str(begin)
        element.attrib['end'] = str(end)
        element.set('departLane', 'best')
        element.set('from', string_from)
        element.set('to', string_to)
        element.attrib['probability'] = str(probability)

        return element

    def _pretty_print(self, current, parent=None, index=-1, depth=0):
        for d, node in enumerate(current):
            self._pretty_print(node, current, d, depth + 1)
        if parent is not None:
            if index == 0:
                parent.text = '\n' + ('\t' * depth)
            else:
                parent[index - 1].tail = '\n' + ('\t' * depth)
            if index == len(parent) - 1:
                current.tail = '\n' + ('\t' * (depth - 1))