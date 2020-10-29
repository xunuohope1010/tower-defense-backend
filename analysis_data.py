import os
import matplotlib.pyplot as plt
import json

generate_list = {}
for folder in os.listdir('./'):
    if 'wave' in folder and '_wave' not in folder:
        generate_list[int(folder[5:])] = len(os.listdir('./' + folder + '/'))

x_generate = []
y_generate = []
max_value_generate = max(generate_list.values())
for key in sorted(generate_list.keys()):
    x_generate.append(key)
    y_generate.append(generate_list[key] / max_value_generate)

verify_list = {}
for folder in os.listdir('./'):
    if '_wave_' in folder:
        verify_list[int(folder[12:])] = len(os.listdir('./' + folder + '/'))

x_verify = []
y_verify = []
max_value_verify = max(verify_list.values())
for key in sorted(verify_list.keys()):
    x_verify.append(key)
    y_verify.append(verify_list[key] / max_value_verify)

plt.plot(x_generate, y_generate, label='before train')
plt.plot(x_verify, y_verify, label='after train')
plt.legend(loc='lower left')
plt.xlabel('Wave')
plt.ylabel('Survival rate')
plt.title('Survival rate: before train VS after train')
plt.show()

r2_list = {}
for file in os.listdir('weight'):
    r2_list[int(file[8:11])] = float(file[12:-5])
print(r2_list)
x_r2 = []
y_r2 = []
for key in sorted(r2_list.keys()):
    x_r2.append(key)
    y_r2.append(r2_list[key])
plt.plot(x_r2, y_r2, label='r2 value')
plt.legend(loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('R2 value')
plt.title('R2 value VS Epoch')
plt.show()

reward_list_generate = []
for folder in os.listdir('./'):
    if 'wave' in folder and '_wave' not in folder:
        for file_name in os.listdir('./'+folder+'/'):
            f = open('./'+folder+'/'+file_name)
            reward_list_generate.append(folder[5:] + ';' + str(json.loads(f.read())['reward']))
            f.close()
print(max_value_generate)
my_dict_generate = {i: reward_list_generate.count(i) / max_value_generate for i in reward_list_generate}
print(my_dict_generate)
x_reward_generate = []
y_reward_generate = []
z_reward_generate = []
for key in sorted(my_dict_generate.keys()):
    arr = key.split(';')
    x_reward_generate.append(int(arr[0]))
    y_reward_generate.append(int(arr[1]))
    z_reward_generate.append(my_dict_generate[key] * 1000)


reward_list_verify = []
for folder in os.listdir('./'):
    if '_wave_' in folder:
        for file_name in os.listdir('./'+folder+'/'):
            f = open('./'+folder+'/'+file_name)
            reward_list_verify.append(folder[12:] + ';' + str(json.loads(f.read())['reward']))
            f.close()
print(max_value_verify)
my_dict_verify = {i: reward_list_verify.count(i) / max_value_verify for i in reward_list_verify}
print(my_dict_verify)
x_reward_verify = []
y_reward_verify = []
z_reward_verify = []
for key in sorted(my_dict_verify.keys()):
    arr = key.split(';')
    x_reward_verify.append(int(arr[0]))
    y_reward_verify.append(int(arr[1]))
    z_reward_verify.append(my_dict_verify[key] * 1000)

plt.scatter(x_reward_generate, y_reward_generate, s=z_reward_generate, label='before training')
plt.scatter(x_reward_verify, y_reward_verify, s=z_reward_verify, label='after training')
plt.legend(loc='lower left')
plt.xlabel('Wave')
plt.ylabel('Reward')
plt.title('Reward VS Wave')
plt.show()
