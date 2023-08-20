from collections import deque

from django.http import JsonResponse, HttpResponse
from django.shortcuts import render

from selenium import webdriver
import tensorflow as tf
'''import argparse
from utils import config'''
from tensorflow import tools

## GPT-2 tf version setting
from tf2 import chatbot_tf2_Output as GPT2_chbot

#EmpDG model
from EmpDG_Interact import interact_out

#ChitChat
from ChitChat_Interact import Chitchat
# 创建Chrome浏览器对象
browser = webdriver.Chrome()
global context
def index(request):
    return render(request, 'chatview.html')


def get_data(request):
    # 处理 Ajax 请求
    if request.method=='POST' and request.is_ajax():
        # 执行 Python 代码，并将结果保存在 result 中
        data = {'name': 'John', 'age': 30, 'gender': 'Male'}
        # 返回 JSON 响应
        return JsonResponse(data)
    elif request.method =='GET':
        # 执行 Python 代码，并将结果保存在 result 中
        data = {'name': 'John', 'age': 30, 'gender': 'Male'}
        # 返回 JSON 响应
        return JsonResponse(data)
    # 渲染 HTML 模板
    return render(request, 'Testpage.html')


#获取页面中输入的文本
def receive_data(request):
    # 处理 Ajax 请求
    if request.method == 'POST' and request.is_ajax():
        user_message = request.POST.get('mydata')
        model_type = request.POST.get('model')
        print('model_type'+model_type)

        if(model_type =='GPT-2'):
            bot_reply = GPT2_chbot.interact_model(user_message,
                nsamples=1,
                top_k=5,
                top_p=1,
                temperature=0.6,
                batch_size=1,
                length=20)
            #bot_reply = 'Chatbot_reply'
            print(user_message)
            data = {'reply': bot_reply}

        elif(model_type =='EmpDG'):
            print("EMPDG")
            bot_reply = interact_out.interact(user_message)
            data = {'reply':bot_reply}
            global context
            context = interact_out.interact(user_message).context
        else:
            print("ChitChat")
            bot_reply = Chitchat.main(user_message)
            data = {'reply':bot_reply}

        return JsonResponse(data)

    elif request.method == 'GET':
        user_message = request.GET.get('mydata')
        model_type = request.GET.get('model')
        print('model_type'+str(model_type))
        if(model_type =='GPT-2'):
            bot_reply = GPT2_chbot.interact_model(user_message,
                nsamples=1,
                top_k=5,
                top_p=1,
                temperature=0.6,
                batch_size=1,
                length=20)
            #bot_reply = 'Chatbot_reply'
            print(user_message)
            data = {'reply': bot_reply}

        elif(model_type =='EmpDG'):
            print("EmpDG")
            bot_reply = interact_out.interact(user_message)
            data = {'reply':bot_reply}

        else:
            print("ChitChat")
            bot_reply = Chitchat.main(user_message)
            data = {'reply': bot_reply}
            # 返回 JSON 响应

        return JsonResponse(data)
    return render(request, 'chatview.html')


def save_data(request):
    data = ['context']
    with open('data.txt', 'w') as f:
        for item in data:
            f.write(str(item) + '\n')
    return JsonResponse({"message": "保存成功"})