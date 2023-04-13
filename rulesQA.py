from transformers import T5Tokenizer, T5ForConditionalGeneration

# model_name = "allenai/unifiedqa-v2-t5-3b-1363200"
model_name = "allenai/unifiedqa-t5-3b" # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return input_ids,tokenizer.batch_decode(res, skip_special_tokens=True)

def run_model2(q, opts=None):
    if (opts):
        ids,ans = run_model(q + ' \\n ' + opts + ' \\n ' + text2)
    else:
        ids,ans = run_model(q + ' \\n ' + text2)
    return ids,ans

def compute_loss(input_ids, answer):
    labels = tokenizer(answer, return_tensors='pt').input_ids
    res = model.forward(input_ids, labels=labels)
    return res.loss.tolist()


def formOpts(opts):
    op2 = ''
    op3 = []
    c = 0
    for i in opts.split(','):
        op2 += '(' + chr(ord('a') + c) + ')' + i.strip() + ' '
        c += 1
        op3.append(i.strip())
    return op2, op3

def getQA(q,ops=None):
    if(ops):
        opts, optlist = formOpts(ops)
        ids, ans = run_model2(q, opts)
        los = compute_loss(ids, ans)
    else:
        ids, ans = run_model2(q)
        los = compute_loss(ids, ans)
    return q,ans,los

while True:
    env = {}

    text2 = input('Input the rule')
    print(text2)
    # Weather Questions
    q,ans,los = getQA('is daytime or nighttime mentioned?', 'yes,no');print(q,ans,los)
    if(ans[0] == 'yes' and los < 0.05):
        q,ans,los = getQA('is it daytime?', 'yes,no');print(q,ans,los)
        if(ans[0] == 'yes' and los < 0.05):
            env['time'] = 'Day'
        q,ans,los = getQA('is it nighttime?', 'yes,no');print(q,ans,los)
        if(ans[0] == 'yes' and los < 0.05):
            env['time'] = 'Night'
    q,ans,los = getQA('is the weather mentioned?', 'yes,no');print(q,ans,los)
    if(ans[0] == 'yes' and los < 0.05):
        q,ans,los = getQA('is a sunny weather mentioned?', 'yes,no');print(q,ans,los)
        if(ans[0] == 'yes' and los < 0.005):
            env['weather'] = 'Sunny'
        q, ans, los = getQA('is a rainy weather mentioned?', 'yes,no');print(q,ans,los)
        if (ans[0] == 'yes' and los < 0.005):
            env['weather'] = 'Rainy'
        q, ans, los = getQA('is a snowy weather mentioned?', 'yes,no');print(q,ans,los)
        if (ans[0] == 'yes' and los < 0.005):
            env['weather'] = 'Snowy'
        q, ans, los = getQA('is a foggy weather mentioned?', 'yes,no');print(q,ans,los)
        if (ans[0] == 'yes' and los < 0.005):
            env['weather'] = 'Foggy'

    road = []
    other = {}
    ego = {}

    #Road Network Questions
    q, ans, los = getQA('is a traffic light mentioned?', 'yes,no');print(q, ans, los)
    if (ans[0] == 'yes' and los < 0.01):
        road.append('Traffic_light')
    else:
        q,ans,los = getQA('is a crosswalk mentioned?', 'yes,no');print(q,ans,los)
        if(ans[0] == 'yes' and los < 0.01):
            road.append('Crosswalk')
        else:
            q, ans, los = getQA('is a roundabout mentioned?', 'yes,no');print(q,ans,los)
            if (ans[0] == 'yes' and los < 0.01):
                road.append('Roundabout')
            else:
                q, ans, los = getQA('is a intersection mentioned?', 'yes,no');print(q,ans,los)
                if (ans[0] == 'yes' and los < 0.01):
                    road.append('Intersection')
                else:
                    q, ans, los = getQA('is a traffic sign explicitly mentioned?', 'yes,no');print(q, ans, los)
                    if (ans[0] == 'yes' and los < 0.01):
                        road.append('Traffic_sign')
                    else:
                        losss = [1,1]
                        q, ans, los = getQA('is it a 1-way street?', 'yes,no');print(q,ans,los)
                        if (ans[0] == 'yes' and los < 0.05):
                            losss[0] = los
                        q, ans, los = getQA('is it a 2-way street?', 'yes,no');print(q,ans,los)
                        if (ans[0] == 'yes' and los < 0.05):
                            losss[1] = los
                        if(losss[0]<1 and losss[1]==1):
                            road.append('One-way-street')
                        elif(losss[0]==1 and losss[1]<1):
                            road.append('Two-way-street')
                        elif (losss[0] < 1 and losss[1] < 1):
                            road.append('Two-way-street')

                        #lanes
                        q, ans, los = getQA('is changing lanes mentioned?', 'yes,no');print(q, ans, los)
                        if (ans[0] == 'yes' and los < 0.01):
                            road.append('Two-lanes')
                        else:
                            q, ans, los = getQA('is it a 1-lane street?', 'yes,no');print(q,ans,los)
                            if (ans[0] == 'yes' and los < 0.01):
                                road.append('One-lane')
                            else:
                                q, ans, los = getQA('is it a 2-lane street?', 'yes,no');print(q, ans, los)
                                if (ans[0] == 'yes' and los < 0.01):
                                    road.append('Two-lanes')

                        #paved
                        q, ans, los = getQA('is it a paved road?', 'yes,no');print(q,ans,los)
                        if (ans[0] == 'yes' and los < 0.01):
                            road.append('Paved-road')
                        else:
                            q, ans, los = getQA('is it a unpaved road?', 'yes,no');print(q,ans,los)
                            if (ans[0] == 'yes' and los < 0.01):
                                road.append('Unpaved-road')

                        #Solidity
                        q, ans, los = getQA('is a solid-line mentioned?', 'yes,no');print(q,ans,los)
                        if (ans[0] == 'yes' and los < 0.01):
                            q, ans, los = getQA('is the solid-line double?', 'yes,no');print(q,ans,los)
                            if (ans[0] == 'yes' and los < 0.01):
                                road.append('Double-solid-line')
                            else:
                                road.append('Solid-line')
                        else:
                            q, ans, los = getQA('is a broken-line mentioned?', 'yes,no');print(q,ans,los)
                            if (ans[0] == 'yes' and los < 0.01):
                                road.append('Broken-line')

    # Weather Questions
    q, ans, los = getQA('is a schoolbus mentioned?', 'yes,no');print(q, ans, los)
    if (ans[0] == 'yes' and los < 0.01):
        other['type'] = 'school-bus'
    else:
        q, ans, los = getQA('is a ambulance mentioned?', 'yes,no');print(q, ans, los)
        if (ans[0] == 'yes' and los < 0.01):
            other['type'] = 'ambulance'
        else:
            q, ans, los = getQA('is a fire_truck mentioned?', 'yes,no');print(q, ans, los)
            if (ans[0] == 'yes' and los < 0.01):
                other['type'] = 'fire_truck'
            else:
                q, ans, los = getQA('is any pedestrian mentioned?', 'yes,no');print(q, ans, los)
                if (ans[0] == 'yes' and los < 0.01):
                    other['type'] = 'pedestrian'
                else:
                    q, ans, los = getQA('is any cyclist mentioned?', 'yes,no');print(q, ans, los)
                    if (ans[0] == 'yes' and los < 0.01):
                        other['type'] = 'cyclist'
                    else:
                        q, ans, los = getQA('is there a car apart from me?', 'yes,no');print(q, ans, los)
                        if (ans[0] == 'yes' and los < 0.05):
                            other['type'] = 'car'


    #my behavior
    q, ans, los = getQA('is changing lanes mentioned?', 'yes,no');print(q, ans, los)
    if (ans[0] == 'yes' and los < 0.05):
        q, ans, los = getQA('am i moving into the right lane?', 'yes,no');print(q, ans, los)
        if (ans[0] == 'yes' and los < 0.01):
            ego['behavior'] = 'change-lane-right'
        q, ans, los = getQA('am i moving into the left lane?', 'yes,no');print(q, ans, los)
        if (ans[0] == 'yes' and los < 0.01):
            ego['behavior'] = 'change-lane-left'
    else:
        q, ans, los = getQA('is turning left or right mentioned?', 'yes,no');print(q, ans, los)
        if (ans[0] == 'yes' and los < 0.05):
            q, ans, los = getQA('is turning left mentioned?', 'yes,no');print(q, ans, los)
            if (ans[0] == 'yes' and los < 0.01):
                ego['behavior'] = 'turn-left'
            q, ans, los = getQA('is turning right mentioned?', 'yes,no');print(q, ans, los)
            if (ans[0] == 'yes' and los < 0.01):
                ego['behavior'] = 'turn-right'
        else:
            q, ans, los = getQA('am i driving?', 'yes,no');print(q, ans, los)
            if (ans[0] == 'yes' and los < 0.05):
                ego['behavior'] = 'travel'
            else:
                q, ans, los = getQA('am i stopped?', 'yes,no');print(q, ans, los)
                if (ans[0] == 'yes' and los < 0.05):
                    ego['behavior'] = 'static'

    # other actor action
    if ('type' in other.keys()):
        q, ans, los = getQA(f'is the {other["type"]} changing lanes?', 'yes,no');print(q, ans, los)
        if (ans[0] == 'yes' and los < 0.05):
            q, ans, los = getQA(f'is the {other["type"]} moving into the right lane?', 'yes,no');print(q, ans, los)
            if (ans[0] == 'yes' and los < 0.01):
                other['behavior'] = 'change-lane-right'
            q, ans, los = getQA(f'is the {other["type"]} moving into the left lane?', 'yes,no');print(q, ans, los)
            if (ans[0] == 'yes' and los < 0.01):
                other['behavior'] = 'change-lane-left'
        else:
            q, ans, los = getQA(f'is the {other["type"]} turning left or right?', 'yes,no');print(q, ans, los)
            if (ans[0] == 'yes' and los < 0.05):
                q, ans, los = getQA(f'is the {other["type"]} turning left?', 'yes,no');print(q, ans, los)
                if (ans[0] == 'yes' and los < 0.01):
                    other['behavior'] = 'turn-left'
                q, ans, los = getQA(f'is the {other["type"]} turning right?', 'yes,no');print(q, ans, los)
                if (ans[0] == 'yes' and los < 0.01):
                    other['behavior'] = 'turn-right'
            else:
                q, ans, los = getQA(f'is the {other["type"]} stopped?', 'yes,no');print(q, ans, los)
                if (ans[0] == 'yes' and los < 0.01):
                    other['behavior'] = 'static'
                else:
                    q, ans, los = getQA(f'is the {other["type"]} driving?', 'yes,no');print(q, ans, los)
                    if (ans[0] == 'yes' and los < 0.01):
                        other['behavior'] = 'travel'

        q, ans, los = getQA(f'is the {other["type"]} answering alarm?', 'yes,no');print(q, ans, los)
        if (ans[0] == 'yes' and los < 0.01):
            if ('behavior') not in other.keys():
                other['behavior'] = 'travel'
            other['behavior'] += '\n answer_alarm'
        else:
            q, ans, los = getQA(f'is the {other["type"]} flashing light?', 'yes,no');print(q, ans, los)
            if (ans[0] == 'yes' and los < 0.01):
                if ('behavior') not in other.keys():
                    other['behavior'] = 'travel'
                other['behavior'] += '\n flash_light'
            else:
                q, ans, los = getQA(f'is the {other["type"]} accelerating?', 'yes,no');print(q, ans, los)
                if (ans[0] == 'yes' and los < 0.01):
                    if ('behavior') not in other.keys():
                        other['behavior'] = 'travel'
                    other['behavior'] += '\n accelerate'
                else:
                    q, ans, los = getQA(f'is the {other["type"]} decelerating?', 'yes,no');print(q, ans, los)
                    if (ans[0] == 'yes' and los < 0.01):
                        if ('behavior') not in other.keys():
                            other['behavior'] = 'travel'
                        other['behavior'] += '\n decelerate'

    #Other actor behavior
    if('type' in other.keys()):
        if('Crosswalk' in road):
            other['position_target'] = 'Crosswalk'
            if(other["type"]=='pedestrian'):
                other['position_relation'] = 'right'
            else:
                q, ans, los = getQA(f'is the {other["type"]} behind the crosswalk?', 'yes,no');print(q, ans, los)
                if (ans[0] == 'yes' and los < 0.01):
                    other['postion_relation'] = 'behind'
        elif ('Intersection' in road):
            other['position_target'] = 'Intersection'
            ego['position_target'] = 'Intersection'
            ego['position_relation'] = 'behind'
            q, ans, los = getQA(f'is the {other["type"]} in the intersection?', 'yes,no');print(q, ans, los)
            if (ans[0] == 'yes' and los < 0.01):
                other['postion_relation'] = 'in'
            else:
                q, ans, los = getQA(f'is the {other["type"]} on the other side of the intersection?', 'yes,no');print(q, ans, los)
                if (ans[0] == 'yes' and los < 0.01):
                    other['postion_relation'] = 'opposite'
        elif ('Roundabout' in road):
            other['position_target'] = 'Roundabout'
            ego['position_target'] = 'Roundabout'
            ego['position_relation'] = 'behind'
            q, ans, los = getQA(f'is the {other["type"]} in the roundabout?', 'yes,no');print(q, ans, los)
            if (ans[0] == 'yes' and los < 0.01):
                other['postion_relation'] = 'in'
        elif ('Traffic_sign' in road):
            other['position_target'] = 'Traffic_sign'
        elif ('Traffic_light' in road):
            other['position_target'] = 'Traffic_sign'
        q, ans, los = getQA(f'is the {other["type"]} driving in front of me?', 'yes,no');print(q, ans, los)
        if (ans[0] == 'yes' and los < 0.01):
            other['postion_relation'] = 'front'
        q, ans, los = getQA(f'is the {other["type"]} driving behind me?', 'yes,no');print(q, ans, los)
        if (ans[0] == 'yes' and los < 0.01):
            other['postion_relation'] = 'behind'
        q, ans, los = getQA(f'is the {other["type"]} driving on my left?', 'yes,no');print(q, ans, los)
        if (ans[0] == 'yes' and los < 0.01):
            if('postion_relation' in other.keys()):
                other['postion_relation'] = 'left ' + other['postion_relation']
            else:
                other['postion_relation'] = 'left'
        q, ans, los = getQA(f'is the {other["type"]} driving on my right?', 'yes,no');print(q, ans, los)
        if (ans[0] == 'yes' and los < 0.01):
            if ('postion_relation' in other.keys()):
                other['postion_relation'] = 'right ' + other['postion_relation']
            else:
                other['postion_relation'] = 'right'


    print(env,road,other,ego)