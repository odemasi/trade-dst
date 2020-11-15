from utils.config import *
from models.MODIFIED_TRADE import *
from copy import deepcopy
from itertools import cycle


import utils.extra_utils as extra_utils



if args['seed'] > -1:
    extra_utils.set_seed(args['seed'])
     
    args['path'] = extra_utils.get_best_model_name('base', args)
    print('Loading base model from: ', args['path'])
    
    # set the name that the newly tuned model should be saved as
    if args['approach'] == 'mixed-meta':
        args['model_name'] = 'mixedmeta'
    elif args['approach'] == 'mixed-multi-task':
        args['model_name'] = 'mixedmultitask'
    
    

except_domain = args['except_domain']
directory = args['path'].split("/")
HDD = directory[-1].split('HDD')[1].split('BSZ')[0]
BSZ = int(args['batch']) if args['batch'] else int(directory[-1].split('BSZ')[1].split('DR')[0])
args["decoder"] = "MODIFIED_TRADE"
args["HDD"] = HDD

if args['dataset']=='multiwoz':
    from utils.utils_multiWOZ_DST import *
else:
    print("You need to provide the --dataset information")

# Load all data on previous domains for testing forgetting
if args['strict_omit']:
    args['allowed_domains'] = [x for x in EXPERIMENT_DOMAINS if x != except_domain]
train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(True, args['task'], False, batch_size=BSZ)


if args['approach'] == 'mixed-meta':
    if args['strict_omit']:
        args['allowed_domains'] = [x for x in EXPERIMENT_DOMAINS if x != except_domain]
    # Load 1% of data from "memory", i.e., previous domains
    args["data_ratio"] = 1
    train_memory,  _, _, _, _, _, _, _ = prepare_data_seq(True, args['task'], False, batch_size=BSZ)
    train_iter = cycle(iter(train_memory))


# Load 1% of data for new slots
if args['strict_omit']:
    args['allowed_domains'] = [except_domain]
args['only_domain'] = except_domain
args['except_domain'] = ''
args["data_ratio"] = 1
train_single, dev_single, test_single, _, _, SLOTS_LIST_single, _, _ = prepare_data_seq(True, args['task'], False, batch_size=BSZ)
args['except_domain'] = except_domain


# Load 1% of all domains
if args['approach'] == 'mixed-multi-task':
    print('Mixed-multi-task tuning!')
    if args['strict_omit']:
        args['allowed_domains'] = EXPERIMENT_DOMAINS
    args['only_domain'] = ''
    args['except_domain'] = ''
    args["data_ratio"] = 1
    train_all, dev_all, test_all, _, _, SLOTS_LIST_all, _, _ = prepare_data_seq(True, args['task'], False, batch_size=BSZ)
    args['except_domain'] = except_domain
    


model = globals()[args["decoder"]](
    int(HDD), 
    lang=lang, 
    path=args['path'], 
    task=args["task"], 
    lr=args["learn"], 
    dropout=args["drop"],
    slots=SLOTS_LIST,
    gating_dict=gating_dict)

avg_best, cnt, acc = 0.0, 0, 0.0
weights_best = deepcopy(model.state_dict())

try:
    for epoch in range(100):
        print("Epoch:{}".format(epoch))  
        # Run the train function
        if args['approach'] == 'mixed-multi-task':
            pbar = tqdm(enumerate(train_all),total=len(train_all))            
        elif args['approach'] == 'mixed-multi-task':
            pbar = tqdm(enumerate(train_single),total=len(train_single))
            
        for i, data in pbar:
        
            if args['approach'] == 'mixed-multi-task':
                model.train_batch(data, int(args['clip']), SLOTS_LIST_all[1], reset=(i==0))
                model.optimize(args['clip'])
            elif args['approach'] == 'mixed-meta':
                data_new = data
                data_old = next(train_iter)
                model.train_meta_batch(data_new, data_old, int(args['clip']), SLOTS_LIST_single[1], SLOTS_LIST[1], reset=(i==0))
            
            pbar.set_description(model.print_loss())

        if((epoch+1) % int(args['evalp']) == 0):
            if args['approach'] == 'mixed-multi-task':
                acc = model.evaluate(dev_all, avg_best, SLOTS_LIST_all[2], args["earlyStop"])
            elif args['approach'] == 'mixed-meta':
                acc = model.evaluate(dev_single, avg_best, SLOTS_LIST_single[2], args["earlyStop"])
                
            model.scheduler.step(acc)
            if(acc > avg_best):
                avg_best = acc
                cnt=0
                weights_best = deepcopy(model.state_dict())    
            else:
                cnt+=1
            if(cnt == args['max_patience'] or (acc==1.0 and args["earlyStop"]==None)): 
                print("Ran out of patient, early stop...")  
                break 
except KeyboardInterrupt: 
    pass

model.load_state_dict({ name: weights_best[name] for name in weights_best })
model.eval()


# After Fine tuning...
print("[Info] After Fine Tune ...")
print("[Info] Test Set on 4 domains...")
acc_test_4d = model.evaluate(test_special, 1e7, SLOTS_LIST[2], return_all=True) 
acc_test_4d['Epochs'] = epoch - cnt # patience counter
json_filename = '/'.join(directory[:-1])+'/results_old_domains_%s.json' % args['model_name']
json.dump(acc_test_4d, open(json_filename, 'w'))


print("[Info] Test Set on 1 domain {} ...".format(except_domain))
acc_test = model.evaluate(test_single, 1e7, SLOTS_LIST[3], return_all=True) 
acc_test['Epochs'] = epoch - cnt # patience counter
json_filename = '/'.join(directory[:-1])+'/results_new_domain_%s.json' % args['model_name']
json.dump(acc_test, open(json_filename, 'w'))



