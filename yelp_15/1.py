f1 = open('/home/hsw/Project/Topic-generation/result_acc_0.8662_acc_real_0.6741/yelp_7.23_no_adv/positive', 'w')
f0 = open('/home/hsw/Project/Topic-generation/result_acc_0.8662_acc_real_0.6741/yelp_7.23_no_adv/negative', 'w')

f = open('/home/hsw/Project/Topic-generation/result_acc_0.8662_acc_real_0.6741/yelp_7.23_no_adv/2_yelp_result_acc_0.9249', 'r')
d = f.readlines()
for i in range(len(d)):
    if i <51207:
        f0.write(d[i].strip())
        f0.write('\n')
    else:
        f1.write(d[i].strip())
        f1.write('\n')