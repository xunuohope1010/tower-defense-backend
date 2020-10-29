import socket
import threading
import json
import random
import numpy as np
import time
import os


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ClientThread(threading.Thread):
    def __init__(self, client_address, client_socket, count_id):
        threading.Thread.__init__(self)
        self.c_socket = client_socket
        self.tower_available = []
        self.tower_map = np.zeros((height, width, 4))

        self.conflict = False
        self.task_list = []
        self.data_list = []
        self.id = count_id

        print("New connection added: ", client_address)

    def generate_one_wave(self):

        place_tower_task = []
        sell_tower_task = []
        upgrade_tower_task = []
        tower_available_copy = self.tower_available.copy()  # for sell only
        # one round, place tower
        number_tower_place = random.randint(min_number_tower_place, max_number_tower_place)

        available_place = path.copy()

        for index_path in path:
            for index_tower in self.tower_available:
                if index_path['x'] == index_tower['x'] and index_path['y'] == index_tower['y']:
                    available_place.remove(index_path)
        if len(available_place) <= 2:
            return None
        index_list = random.sample(available_place, number_tower_place)
        for index in index_list:
            tower_list = [0, 0, 0, 0]
            tower_index = random.randint(0, 3)
            tower_list[tower_index] = 1
            self.tower_available.append({'x': index['x'], 'y': index['y'], 'tower_list': tower_list})
            # add to map
            self.tower_map[index['y']][index['x']][tower_index] = 1
            # add to task
            place_tower_task.append({'x': index['x'], 'y': index['y'], 'type': tower_index + 1})

        # print("after place tower")
        # print(self.tower_available)
        # one round, sell tower

        number_tower_sell = random.randint(0, max_number_tower_sell)

        if len(tower_available_copy) < number_tower_sell:
            number_tower_sell = len(tower_available_copy)
        index_list = random.sample(tower_available_copy, number_tower_sell)
        for index in index_list:
            # update the map
            self.tower_map[index['y']][index['x']] = [0, 0, 0, 0]
            # add to task
            sell_tower_task.append({'x': index['x'],
                                    'y': index['y']})
            # remove from list
            # print("remove index")
            # print(index)
            self.tower_available.remove(index)

        # print("after sell tower")
        # print(self.tower_available)
        # one round, upgrade tower

        number_tower_upgrade = random.randint(0, max_number_tower_upgrade)
        if len(self.tower_available) < number_tower_upgrade:
            number_tower_upgrade = len(self.tower_available)
        index_list = random.sample(self.tower_available, number_tower_upgrade)
        for index in index_list:
            upgrade_level = random.randint(1, max_upgrade_level)
            tower_list = index['tower_list']
            # print("tower_list")
            # print(tower_list)
            for pointer in range(len(tower_list)):
                if tower_list[pointer] == 1:
                    tower_list[pointer] += upgrade_level
                    self.tower_map[index['y']][index['x']][pointer] = tower_list[pointer]
                    break
            for pointer in range(len(self.tower_available)):
                if self.tower_available[pointer] == index:
                    self.tower_available[pointer]['tower_list'] = tower_list
            upgrade_tower_task.append({'x': index['x'], 'y': index['y'], 'level': upgrade_level})

        # print("after upgrade tower")
        # print(self.tower_available)

        task = {'place_tower': place_tower_task, 'sell_tower': sell_tower_task, 'upgrade_tower': upgrade_tower_task}
        # self.layers.append(self.tower_map)
        return task

    def run(self):
        print("Connection from : ", clientAddress)
        init = True
        while True:
            if init:
                para = {'wave_round': wave_round, 'init_money': init_money, 'init_lives': init_lives,
                        'time_left': time_left, 'game_map': lines, 'start': start, 'end': end,
                        'orthographic_size': orthographic_size, 'time_scale': time_scale,
                        'time_one_wave': time_one_wave}
                self.c_socket.send(json.dumps(para).encode('utf8'))
                init = False

            data = self.c_socket.recv(2048)

            if not data:
                break

            received = data.decode('utf8')
            if 'ok' in received:
                try:
                    get_json = json.loads(received)
                    min_path_length = get_json['min_path_length']
                    lives = get_json['lives']
                    reward = get_json['reward']
                    wave_number = get_json['wave_number']
                    self.data_list.append({'game_map': self.tower_map, 'lives': lives, 'wave_number': wave_number,
                                           'min_path_length': min_path_length, 'reward': reward})

                    # print(received)
                except ValueError:
                    print("first round")
                task = self.generate_one_wave()
                if not task:
                    self.c_socket.send('stop'.encode('utf8'))
                    print("send stop, reason no place for tower!")
                    continue
                print(json.dumps(task))

                self.c_socket.send(json.dumps(task).encode('utf8'))

            if 'stop' in received:

                # game_map = np.stack(self.layers)
                try:
                    get_json = json.loads(received)

                except ValueError:
                    print("game internal error! skip!")
                    self.c_socket.close()
                    break

                for my_data in self.data_list:
                    folder_name = 'wave_' + str(my_data['wave_number'])
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)

                    my_file = open('./' + folder_name + '/' + str(self.id) + '.txt', 'w')
                    my_file.write(json.dumps(my_data, cls=NumpyEncoder))
                    my_file.close()
                self.c_socket.close()

                break
            pass
        pass


# set parameters here, for each wave
max_number_tower_place = 6
min_number_tower_place = 2
max_number_tower_sell = 2
max_number_tower_upgrade = 2
max_upgrade_level = 2
wave_round = 5  # disabled in game
init_money = 500
init_lives = 50
time_left = 4200  # disabled in game
time_one_wave = 800
orthographic_size = 5
time_scale = 5
start = {'x': 0, 'y': 0}
end = {'x': 11, 'y': 6}
width = 12
height = 8
max_number_sample = 3000

f = open('game_map.txt', 'r')
lines = []
matrix = []
while True:
    line = f.readline()
    if line == '':
        break
    matrix.append(list(line[:-1]))
    lines.append(line[:-1])

print(matrix)
path = []
for i in range(0, height):
    for j in range(0, width):

        if matrix[i][j] == '1':
            path.append({'x': j, 'y': i})
print(path)

# host parameters
LOCALHOST = "127.0.0.1"
PORT = 8081
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((LOCALHOST, PORT))
print("Server started")
print("Waiting for client request..")
count = 0
while True:
    if count >= max_number_sample:
        break
    server.listen(1)
    client_sock, clientAddress = server.accept()
    new_thread = ClientThread(clientAddress, client_sock, count)
    new_thread.start()
    count += 1
