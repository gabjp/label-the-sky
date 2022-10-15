from label_the_sky.training.trainer import Trainer
import os
import sys

if len(sys.argv) != 8:
    print('usage: python {} <dataset> <backbone> <pretraining_dataset> <n_channels> <finetune> <dataset_mode> <timestamp>'.format(
        sys.argv[0]))
    exit(1)

dataset = sys.argv[1]
backbone = sys.argv[2]
pretraining_dataset = None if sys.argv[3]=='None' else sys.argv[3]
n_channels = int(sys.argv[4])
finetune = True if sys.argv[5]=='1' else False
dataset_mode = sys.argv[6]
timestamp = sys.argv[7]

model_name = f'{timestamp}_{backbone}_{n_channels}_{pretraining_dataset}_clf_ft{int(finetune)}_{dataset_mode}'
base_dir = os.environ['HOME']

trainer = Trainer(
    backbone=backbone,
    n_channels=n_channels,
    output_type='class',
    base_dir=base_dir,
    weights=os.path.join(base_dir, 'trained_models', model_name+'.h5'),
    model_name=model_name
)

print('loading data')
X_train, y_train = trainer.load_data(dataset=dataset, split='train')
X_val, y_val = trainer.load_data(dataset=dataset, split='val')
X_test, y_test = trainer.load_data(dataset=dataset, split='test')

print('evaluating model on validation set')
trainer.evaluate(X_val, y_val)

print('evaluating model on test set')
trainer.evaluate(X_test, y_test)

