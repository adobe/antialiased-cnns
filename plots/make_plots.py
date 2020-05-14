
import matplotlib.pyplot as plt
import numpy as np

taps = [1,2,3,5]

accs = {}
accs['alexnet'] = [56.55, 57.24, 56.90, 56.58]
accs['vgg16'] = [71.59, 72.15, 72.20, 72.33]
accs['vgg16bn'] = [73.36, 74.01, 73.91, 74.05]
accs['resnet18'] = [69.74, 71.39, 71.69, 71.38]
accs['resnet34'] = [73.30, 74.46, 74.33, 74.20]
accs['resnet50'] = [76.16, 76.81, 76.83, 77.04]
accs['resnet101'] = [77.37, 77.82, 78.13, 77.92]
accs['densenet121'] = [74.43, 75.04, 75.14, 75.03]
accs['mobilenet'] = [71.88, 72.63, 72.59, 72.50]

cons = {}
cons['alexnet'] = [78.18, 81.33, 82.15, 82.51]
cons['vgg16'] = [88.52, 89.24, 89.60, 90.19]
cons['vgg16bn'] = [89.24, 90.72, 91.10, 91.35]
cons['resnet18'] = [85.11, 86.90, 87.51, 88.25]
cons['resnet34'] = [87.56, 89.14, 89.32, 89.49]
cons['resnet50'] = [89.20, 89.96, 90.91, 91.31]
cons['resnet101'] = [89.81, 91.04, 91.62, 91.74]
cons['densenet121'] = [88.81, 89.53, 89.78, 90.39]
cons['mobilenet'] = [86.50, 87.33, 87.46, 87.79]

mins = {}
mins['alexnet'] = [56, 78]
mins['vgg16'] = [71, 88]
mins['vgg16bn'] = [73, 89]
mins['resnet18'] = [69, 85]
mins['resnet34'] = [73, 87]
mins['resnet50'] = [76, 89]
mins['resnet101'] = [77, 89]
mins['densenet121'] = [75, 88]
mins['mobilenet'] = [71, 86]

names = dict(alexnet='AlexNet',
	vgg16='VGG16',
	vgg16bn='VGG16bn',
	resnet18='ResNet18',resnet34='ResNet34',resnet50='ResNet50',resnet101='ResNet101',
	densenet121='DenseNet121',
	mobilenet='MobileNetv2')

labels = ['Baseline','Rect-2','Tri-3','Bin-5']

chars = ['','o',[(-.125,-.5),(.125,-.5),(.125,.5),(-.125,.5),(-.125,-.5)]
	,'^','d','p','h',(7,0,0)] # by filter size
keys = ['alexnet', 'vgg16', 'vgg16bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'densenet121', 'mobilenet']


fills = ['k','w','w','w']
sizes = [6,12,10,12]

colors = dict(gray='#4D4D4D', blue='#5DA5DA', orange='#FAA43A', green='#60BD68', pink='#F17CB0', brown='#B2912F', purple='#B276B2', yellow='#DECF3F', red='#F15854')
net_colors = dict(alexnet=colors['red'], vgg16=colors['red'], vgg16bn=colors['red'], 
	resnet18=colors['blue'], resnet34=colors['blue'], resnet50=colors['blue'], resnet101=colors['blue'],
	densenet121=colors['pink'],
	mobilenet=colors['purple'])


# Plot everything besides AlexNet
sizes = [6,9,7.5,9]
plt.figure(figsize=(6,5))
for (kk,key) in enumerate(keys[1:]):
	plt.plot(accs[key], cons[key], linestyle='-', color=net_colors[key])
	for tt, tap in enumerate(taps):
		plt.plot(accs[key][tt], cons[key][tt], linestyle='', marker=chars[tap], color=net_colors[key], 
			markersize=sizes[tt], markerfacecolor='w' if tt>0 else net_colors[key])
	plt.xlabel('Accuracy')
	plt.plot(0, 0, marker='o', linestyle='', color='k', markerfacecolor='w')
plt.ylabel('Consistency')
plt.plot(0, 0, 'o', markersize=6, markeredgecolor='k', markerfacecolor='k', label='Baseline')
for tt, tap in enumerate(taps[1:]):
	plt.plot(0, 0, linestyle='', marker=chars[tap], color='k', 
		markersize=sizes[tt+1], markerfacecolor=fills[tt+1], label='Anti-aliased (%s)'%labels[tt+1])
plt.legend(loc=4, fontsize='small', labelspacing=.5, ncol=1)
plt.text(71.7, 88.45, 'VGG16', verticalalignment='top', horizontalalignment='left',fontsize='medium', color=net_colors['vgg16'])
plt.text(72.4, 89.0, 'VGG16bn', verticalalignment='top', horizontalalignment='left',fontsize='medium', color=net_colors['vgg16bn'])
plt.text(74.4, 88.65, 'DenseNet121', verticalalignment='top', horizontalalignment='left',fontsize='medium', color=net_colors['densenet121'])
plt.text(69.4, 84.90, 'ResNet18', verticalalignment='top', horizontalalignment='left',fontsize='medium', color=net_colors['resnet18'])
plt.text(73.45, 87.5, 'ResNet34', verticalalignment='top', horizontalalignment='left',fontsize='medium', color=net_colors['resnet34'])
plt.text(76.4, 89.3, 'ResNet50', verticalalignment='top', horizontalalignment='left',fontsize='medium', color=net_colors['resnet50'])
plt.text(77.6, 90.0, 'ResNet101', verticalalignment='top', horizontalalignment='left',fontsize='medium', color=net_colors['resnet101'])
plt.text(71.4, 86.3, 'Mobilenet-v2', verticalalignment='top', horizontalalignment='left',fontsize='medium', color=net_colors['mobilenet'])
plt.xlim((69, 80))
plt.ylim((84, 93))
plt.savefig('imagenet_ind2_noalex.pdf',bbox_inches='tight')
plt.savefig('imagenet_ind2_noalex.jpg',bbox_inches='tight',dpi=750)
plt.close()



# Individual plots. Each is a figure
for (kk,key) in enumerate(keys):
	plt.figure(figsize=(3,3))
	for tt, tap in enumerate(taps):
		plt.plot(accs[key][tt], cons[key][tt], linestyle='', marker=chars[tap], color='k', markersize=sizes[tt], markerfacecolor=fills[tt], label=labels[tt])
	plt.xlim((np.round(2*accs[key][0]-1.)/2., np.round(2*accs[key][0]-1.)/2.+3. ))
	plt.ylim((np.round(cons[key][0]-1.5), np.round(cons[key][0]-1.5)+6. ))
	plt.title(names[key])
	plt.xlabel('Accuracy')
	plt.legend(loc=4, fontsize='x-small', labelspacing=1)
	plt.ylabel('Consistency')
	plt.savefig('imagenet_ind_%s.pdf'%names[key],bbox_inches='tight')
	plt.savefig('imagenet_ind_%s.jpg'%names[key],bbox_inches='tight')
plt.close()



sizes = [6,10,8,10]

plt.figure(figsize=(5,4))
for (kk,key) in enumerate(keys):
	for tt, tap in enumerate(taps):
		plt.plot(accs[key][tt], cons[key][tt], linestyle='', marker=chars[tap], color=net_colors[key], 
			markersize=sizes[tt], markerfacecolor='None' if tt>0 else net_colors[key])
	plt.xlabel('Accuracy')
	plt.plot(0, 0, marker='o', linestyle='', color=net_colors[key], markerfacecolor='w', label=names[key])
plt.ylabel('Consistency')
plt.plot(0, 0, 'o', markersize=6, markeredgecolor='k', markerfacecolor='k', label='Baseline')
for tt, tap in enumerate(taps[1:]):
	plt.plot(0, 0, linestyle='', marker=chars[tap], color='k', 
		markersize=sizes[tt+1], markerfacecolor=fills[tt+1], label='Anti-aliased (%s)'%labels[tt+1])
plt.legend(loc=4, fontsize='small', labelspacing=.5, ncol=2)
plt.xlim((55, 78))
plt.ylim((77, 93))
plt.savefig('imagenet_agg_all2.pdf',bbox_inches='tight')
# plt.show()


sizes = [6,10,8,10]

plt.figure(figsize=(5,4))
for (kk,key) in enumerate(keys):
	# plt.plot(accs[key], cons[key], '-')
	for tt, tap in enumerate(taps):
		plt.plot(accs[key][tt], cons[key][tt], linestyle='', marker=chars[tap], color=net_colors[key], 
			markersize=sizes[tt], markerfacecolor='None' if tt>0 else net_colors[key])
	plt.xlabel('Accuracy')
plt.ylabel('Consistency')
plt.plot(0, 0, 'o', markersize=6, markeredgecolor='k', markerfacecolor='k', label='Baseline')
for tt, tap in enumerate(taps[1:]):
	plt.plot(0, 0, linestyle='', marker=chars[tap], color='k', 
		markersize=sizes[tt+1], markerfacecolor=fills[tt+1], label='Anti-aliased (%s)'%labels[tt+1])
plt.text(57.1, 78.18, 'AlexNet', verticalalignment='center', horizontalalignment='left',fontsize='small', color=net_colors['alexnet'])
plt.text(70.6, 88.1, 'VGG16', verticalalignment='top', horizontalalignment='left',fontsize='small', color=net_colors['vgg16'])
plt.text(74.03, 88.45, 'DenseNet121', verticalalignment='top', horizontalalignment='left',fontsize='small', color=net_colors['densenet121'])
plt.text(76.60, 89.22, 'ResNet50', verticalalignment='top', horizontalalignment='left',fontsize='small', color=net_colors['resnet50'])
plt.legend(loc=4, fontsize='small', labelspacing=.5, ncol=1)
plt.xlim((55, 81))
plt.ylim((77, 93))
plt.savefig('imagenet_agg_all_line2_colored.pdf',bbox_inches='tight')


plt.close()
plt.figure(figsize=(5,4))
for (kk,key) in enumerate(keys):
	plt.plot(np.array(accs[key])[[0,3]]/100., np.array(cons[key])[[0,3]]/100.,'-',color=net_colors[key], label=names[key])
	plt.plot(np.array(accs[key])[0]/100., np.array(cons[key])[0]/100.,'o', color=net_colors[key], markersize=6, markerfacecolor=net_colors[key])
	plt.plot(np.array(accs[key])[3]/100., np.array(cons[key])[3]/100.,'p', color=net_colors[key], markersize=10, markerfacecolor='w')
	# plt.plot(np.array(accs[key])[2]/100., np.array(cons[key])[2]/100.,'^', color=net_colors[key], markersize=10, markerfacecolor='w')
plt.plot(0, 0, 'o', markersize=6, markeredgecolor='k', markerfacecolor='k', label='Baseline')
plt.plot(0, 0, '^', markersize=10, markeredgecolor='k', markerfacecolor='w', label='Ours (Tri-3 filter)')
# plt.legend(loc=4,ncol=1,fontsize='small')
plt.xlim((.55, .80))
plt.ylim((.76, .93))
plt.xlabel('Accuracy')
plt.ylabel('Consistency')
plt.savefig('imagenet_agg2.pdf',bbox_inches='tight')
# plt.show()


# **** print table ****
for tt in range(4):
	print('{\\bf %s}'%labels[tt])
	for key in ['alexnet','vgg16','vgg16bn']:
		print_str = '& '
		if(np.argmax(accs[key])==tt):
			print_str += '{\\bf %.1f}'%accs[key][tt]
		else:
			print_str += '%.1f'%accs[key][tt]
		if(np.argmax(cons[key])==tt):
			print_str += '& {\\bf %.1f}'%cons[key][tt]
		else:
			print_str += '& %.1f'%cons[key][tt]
		print(print_str)
	print('\\\\')
	if(tt==0):
		print('\\cdashline{1-9}')


# **** print big table ****
for tt in range(4):
	print('{\\bf %s}'%labels[tt])
	for key in ['resnet18','resnet34','resnet50','resnet101',]:
		print_str = '& '
		acc_diff = (accs[key][tt]-accs[key][0])
		con_diff = (cons[key][tt]-cons[key][0])
		acc_diff_str = '%.2f'%acc_diff if tt>0 else '--'
		con_diff_str = '%.2f'%con_diff if tt>0 else '--'
		acc_pm_str = '+' if acc_diff else ''
		con_pm_str = '+' if con_diff else ''
		if(np.argmax(accs[key])==tt):
			print_str += '{\\bf %.2f} & {\\bf %s%s} '%(accs[key][tt], acc_pm_str, acc_diff_str)
		else:
			print_str += '%.2f & %s%s '%(accs[key][tt], acc_pm_str, acc_diff_str)
		if(np.argmax(cons[key])==tt):
			print_str += '& {\\bf %.2f} & {\\bf %s%s} '%(cons[key][tt], con_pm_str, con_diff_str)
		else:
			print_str += '& %.2f & %s%s '%(cons[key][tt], con_pm_str, con_diff_str)
		print(print_str)
	print('\\\\')
	if(tt==0):
		print('\\cdashline{1-17}')


# **** Print github tables ****
for key in ['alexnet','vgg16','vgg16bn','resnet18','resnet34','resnet50','resnet101','densenet121','mobilenet']:
	print('**%s**'%names[key])
	print('|          | Accuracy | Consistency |')
	print('| :------: | :------: | :---------: |')
	for tt in range(4):
		print('| %s | %.2f | %.2f | '%(labels[tt],accs[key][tt], cons[key][tt]))
	print('')


