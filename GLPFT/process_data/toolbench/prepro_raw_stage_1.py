import json
import argparse
import os
import re
import tqdm
# this code will reorganize the training data of toolbench into the conversation form, and change to tool document schema

print("####################### PREPRO RAW DATA STAGE 1 #####################")

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="") #data_dir 获得了$ORI_DATA_DIR的路径 "../data/toolbench/data"
parser.add_argument('--output_path', type=str, default='') #output_path 获得了RAW_DATA_OUT_DIR的路径 "dataset/toolbench/train/raw_data"

args = parser.parse_args()


test_ids = []
for test_set in ['G1_category', 'G1_instruction', 'G1_tool', 'G2_category', 'G2_instruction', 'G3_instruction']:
    test_ids += list(json.load(open(f"{args.data_dir}/test_query_ids/{test_set}.json")))

print('test_ids', test_ids)

data = []
for train_set in ["G1_answer", "G2_answer", "G3_answer"]:
    for file in tqdm.tqdm(os.listdir(f"{args.data_dir}/answer/{train_set}")): #tqdm加了进度条
        id = file.split('_')[0]
        if id not in test_ids:
            with open(f"{args.data_dir}/answer/{train_set}/{file}") as f:
                d = json.load(f)   #d是文件夹G1/G2/G3 answer中一个json文件的内容
            instruction = d['answer_generation']['query'] #一个json文件一个query
            new_tools = []
            for t in d['answer_generation']['function']: #t是'function'下的子函数
                # tool_name = change_name(standardize(t['api_name'])) + '_for_' + standardize(t['tool_name'])
                tool_name = t['name']
                if tool_name != "Finish":   #当name=finish的时候，才是结束这个函数的调用
                    tool_name = tool_name[-64:]
                    tool_function = t['description'][:256]
                    tool_input = {}
                    for arg_name,arg_value in t['parameters']['properties'].items(): #子函数-parameters-properties-（参数，值）
                        arg_type = arg_value['type']      
                        if 'description' in arg_value.keys():  #检查当前遍历的参数值中是否包含description键
                            tool_input[arg_name] = arg_type + ', ' + arg_value['description'][-256:]
                        else:
                            tool_input[arg_name] = arg_type
                    # print(json.dumps(t, indent=2)) #将t转化为json格式
                    new_tools.append({
                        'Name': tool_name,
                        'function':tool_function,
                        'input':tool_input
                    })
            data.append({
                'tools':new_tools,
                'instruction':instruction,
                'chains':d['answer_generation']['train_messages'][-1] #-1是取最后一次训练的回答。每次train_message都循序渐进，没调用-开始调用-问询-函数查询结果
            })
# instruciton 是answer_generation中的query，chain是训练得到的最终回答
# data中是  工具-问询-训练后获得的回答 的列表
    print(train_set,'data processed')
            
os.makedirs(args.output_path, exist_ok=True)
with open(f"{args.output_path}/raw_data_stage_1.json", 'w') as f:
    json.dump(data[:20],f, indent=2)
