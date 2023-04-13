from tkinter.filedialog import askopenfilename

from transformers import T5Tokenizer, T5ForConditionalGeneration

# model_name = "allenai/unifiedqa-v2-t5-3b-1251000" # you can specify the model size here
model_name = "allenai/unifiedqa-t5-3b" # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return input_ids,tokenizer.batch_decode(res, skip_special_tokens=True)


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
    return op2,op3

# filename = askopenfilename()
# print(filename)
# # with open(baseDir+'Natural08.xml') as sFile:
# with open(filename) as sFile:
#     text = sFile.read()
# print(text)
text = 'When changing lanes, you must give way to vehicles in the lane you’re moving into.'

q = ''

while(True):
    print(text)
    q = input("Enter question:(ext for termination)")
    if(q == 'ext'):
        break

    opts = input("Enter options")
    if(opts):
        op2,op3 = formOpts(opts)
        print(op2)
        ids,ans = run_model(q+' \\n '+op2+' \\n '+text)
    else:
        ids,ans = run_model(q + ' \\n ' + text)
    print(type(ans))
    print(ans)
    print(compute_loss(ids,ans))
