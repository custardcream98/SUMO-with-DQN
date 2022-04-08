from xml.etree.ElementTree import Element, SubElement, ElementTree
import random


# 6 to 4 / 4 to 6
HORIZONTAL_STRAIGHT_CAPACITY = [0.3,0.3,0.24,0.24,0.24,0.18,0.18,0.18,0.3,0.3,0.24,0.24,0.24,0.18,0.18,0.18]
# 6 to 5 / 4 to 7
HORIZONTAL_LEFT_RATIO = [0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06]
# 5 to 7 / 7 to 5
VERTICAL_STRAIGHT_CAPACITY = [0.24,0.18,0.3,0.24,0.18,0.3,0.24,0.18,0.24,0.18,0.3,0.24,0.18,0.3,0.24,0.18]
# 5 to 4 / 7 to 6
VERTICAL_LEFT_RATIO = [0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03]


CAP_AND_RATIOS = [
    HORIZONTAL_STRAIGHT_CAPACITY,
    HORIZONTAL_LEFT_RATIO,
    VERTICAL_STRAIGHT_CAPACITY,
    VERTICAL_LEFT_RATIO,
]

DIRECTIONS = [
    ["-gneE6", "gneE4"], # h-s
    ["-gneE4", "gneE6"], # h-s
    ["-gneE6", "gneE5"], # h-l
    ["-gneE4", "gneE7"], # h-l
    ["-gneE5", "gneE7"], # v-s
    ["-gneE7", "gneE5"], # v-s
    ["-gneE5", "gneE4"], # v-l
    ["-gneE7", "gneE6"], # v-l
]

def _pretty_print(current, parent=None, index=-1, depth=0):
    for i, node in enumerate(current):
        _pretty_print(node, current, i, depth + 1)
    if parent is not None:
        if index == 0:
            parent.text = '\n' + ('\t' * depth)
        else:
            parent[index - 1].tail = '\n' + ('\t' * depth)
        if index == len(parent) - 1:
            current.tail = '\n' + ('\t' * (depth - 1))


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

for i in range(0, len(HORIZONTAL_STRAIGHT_CAPACITY)):
    while True:
        current_case = random.randint(0, 15)
        if current_case not in case_made:
            case_made.append(current_case)
            break

    begin = i * 1800 + 1 if i != 0 else 0
    end = (i + 1) * 1800

    for j in range(0, 4):
        element_a = Element('flow')
        element_b = Element('flow')

        element_a.set('id', f'HV{i * 8 + j * 2 + 1:03}')
        element_b.set('id', f'HV{i * 8 + j * 2 + 2:03}')

        element_a.set('type', 'Krauss')
        element_a.attrib['begin'] = str(begin)
        element_a.attrib['end'] = str(end)
        element_a.set('departLane', 'best')

        element_b.set('type', 'Krauss')
        element_b.attrib['begin'] = str(begin)
        element_b.attrib['end'] = str(end)
        element_b.set('departLane', 'best')

        element_a.set('from', DIRECTIONS[2*j][0])
        element_a.set('to', DIRECTIONS[2*j][1])
        element_b.set('from', DIRECTIONS[2*j + 1][0])
        element_b.set('to', DIRECTIONS[2*j + 1][1])

        element_a.attrib['probability'] = str(CAP_AND_RATIOS[j][case_made[i]])
        element_b.attrib['probability'] = str(CAP_AND_RATIOS[j][case_made[i]])

        root.append(element_a)
        root.append(element_b)

_pretty_print(root)
print(root)
tree = ElementTree(root)

fileName = "intersection_test.rou.xml"
with open(fileName, "wb") as file:
    tree.write(file, encoding='utf-8', xml_declaration=False)