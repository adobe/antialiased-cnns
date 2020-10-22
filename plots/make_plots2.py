
import matplotlib.pyplot as plt
import numpy as np

colors = dict(gray='#4D4D4D', blue='#5DA5DA', orange='#FAA43A', green='#60BD68', pink='#F17CB0', brown='#B2912F', purple='#B276B2', yellow='#DECF3F', red='#F15854')

names = {}
names['alexnets'] = ['alexnet']
names['vggs'] = ['VGG11','VGG13','VGG16','VGG19']
names['vggbns'] = ['VGG11bn','VGG13bn','VGG16bn','VGG19bn']
names['resnets'] = ['Resnet18','Resnet34','Resnet50','Resnet101','Resnet152']
names['resnexts'] = ['Resnext50','Resnext101']
names['wideresnets'] = ['WideResnet50','WideResnet101']
names['densenets'] = ['Densenet121','Densenet169','Densenet201','Densenet161']
names['mobilenets'] = ['Mobilenetv2']

accs_base = {}
accs_base['alexnets'] = [56.55,]
accs_base['vggs'] = [69.02, 69.93, 71.59, 72.38]
accs_base['vggbns'] = [70.38, 71.55, 73.36, 74.24]
accs_base['resnets'] = [69.74, 73.3, 76.16, 77.37, 78.31]
accs_base['resnexts'] = [77.62, 79.31]
accs_base['wideresnets'] = [78.47, 78.85]
accs_base['densenets'] = [74.43, 75.60, 76.90, 77.14]
accs_base['mobilenets'] = [71.88,]

accs_aa = {}
accs_aa['alexnets'] = [56.94,]
accs_aa['vggs'] = [70.51, 71.52, 72.96, 73.54]
accs_aa['vggbns'] = [72.63, 73.61, 75.13, 75.68]
accs_aa['resnets'] = [71.67, 74.60, 77.41, 78.38, 79.07]
accs_aa['resnexts'] = [77.93, 79.33]
accs_aa['wideresnets'] = [78.70, 78.99]
accs_aa['densenets'] = [75.79, 76.73, 77.31, 77.88]
accs_aa['mobilenets'] = [72.72, ]

cons_base = {}
cons_base['alexnets'] = [78.18,]
cons_base['vggs'] = [86.58, 86.92, 88.52, 89.17]
cons_base['vggbns'] = [87.16, 88.03, 89.24, 89.59]
cons_base['resnets'] = [85.11, 87.56, 89.20, 89.81, 90.92]
cons_base['resnexts'] = [90.17, 91.33]
cons_base['wideresnets'] = [90.77, 90.93]
cons_base['densenets'] = [88.81, 89.68, 90.36, 90.82]
cons_base['mobilenets'] = [86.50, ]

cons_aa = {}
cons_aa['alexnets'] = [83.31,]
cons_aa['vggs'] = [90.09, 90.31, 90.91, 91.08]
cons_aa['vggbns'] = [90.666, 91.089, 91.580, 91.597, ]
cons_aa['resnets'] = [88.356, 89.766, 91.316, 91.966, 92.424, ]
cons_aa['resnexts'] = [91.48, 92.67]
cons_aa['wideresnets'] = [92.46, 92.10]
cons_aa['densenets'] = [90.35, 90.61, 91.32, 91.66]
cons_aa['mobilenets'] = [87.733, ]


ticks_x = []
ticks_lbl = []
fig = plt.figure(figsize=(10,5))
plt.ion()
cnt = 0
for key in ['vggs', 'vggbns', 'mobilenets', 'resnets', 'densenets', 'wideresnets', 'resnexts',]:
	for a in range(len(accs_base[key])):
		plt.bar(cnt-.2, accs_base[key][a], width=.4, color=colors['red'])
		plt.bar(cnt+.2, accs_aa[key][a], width=.4, color=colors['blue'])
		ticks_x.append(cnt)
		ticks_lbl.append(names[key][a])
		cnt+=1
	cnt+=.5
plt.bar(-5, 0, color=colors['red'], label='Baseline')
plt.bar(-5, 0, color=colors['blue'], label='Antialiased')
plt.xlim((-1, cnt))
plt.ylim((68,80))
plt.xticks(ticks_x, ticks_lbl, rotation=30, ha='right', va='top')
plt.title('Accuracy (Baseline --> Antialiased)')
plt.ylabel('Accuracy')
plt.legend(loc=2)
fig.tight_layout()
# plt.show()
plt.savefig('plots/plots2_acc.png')

ticks_x = []
ticks_lbl = []
fig = plt.figure(figsize=(10,5))
plt.ion()
cnt = 0
for key in ['vggs', 'vggbns', 'mobilenets', 'resnets', 'densenets', 'wideresnets', 'resnexts',]:
	for a in range(len(cons_base[key])):
		plt.bar(cnt-.2, cons_base[key][a], width=.4, color=colors['red'])
		plt.bar(cnt+.2, cons_aa[key][a], width=.4, color=colors['blue'])
		ticks_x.append(cnt)
		ticks_lbl.append(names[key][a])
		cnt+=1
	cnt+=.5
plt.bar(-5, 0, color=colors['red'], label='Baseline')
plt.bar(-5, 0, color=colors['blue'], label='Antialiased')
plt.xlim((-1, cnt))
plt.ylim((84,93))
plt.xticks(ticks_x, ticks_lbl, rotation=30, ha='right', va='top')
plt.ylabel('Consistency')
plt.title('Consistency (Baseline --> Antialiased)')
plt.legend(loc=2)
fig.tight_layout()
plt.savefig('plots/plots2_con.png')
# plt.show()

