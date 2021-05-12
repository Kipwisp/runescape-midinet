import argparse
import tensorflow as tf
import model.model as models
import utils.dataset as dataset_utils
from datetime import datetime

class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, manager, best_manager):
        super()
        self.manager = manager
        self.best_manager = best_manager
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs):
        if logs['val_loss'] < self.best_loss:
            print('New best val_loss!')
            self.best_loss = logs['val_loss']
            self.best_manager.save()
        
        self.manager.save()
        
        print(f'Current best val_loss: {self.best_loss:.4f}')


def train(num_epochs, save_dir, dataset, restore):
    data = dataset_utils.load_dataset(dataset)

    split = 0.1
    train_split, validation_split, batch_size = dataset_utils.create_generator_and_validation(data, split)
    model, optimizer, loss_function = models.create_model()

    max_to_keep = 25
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory='checkpoints', max_to_keep=max_to_keep)

    best_checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    best_manager = tf.train.CheckpointManager(best_checkpoint, directory='checkpoints/best', max_to_keep=1)

    log_dir = f'logs/{datetime.now().strftime("%d-%b-%Y_%H.%M.%S")}'

    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=num_epochs*0.1),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False),
        CheckpointCallback(manager, best_manager)
    ]

    if restore:
        print('Restoring from latest checkpoint.')
        manager.restore_or_initialize()

    model.fit(x=train_split, epochs=num_epochs, batch_size=batch_size, validation_data=validation_split, callbacks=callbacks)

    print(f'TensorBoard logs can be found under \'{log_dir}\'.')

    best_manager.restore_or_initialize()
    print(f'Saving model as \'{save_dir}\'.')
    model.save(save_dir)


def main():
    parser = argparse.ArgumentParser(description='Train the model on a dataset.')
    parser.add_argument('-e', '--epochs', type=int, help='the number of epochs to train for')
    parser.add_argument('-s', '--save_directory', default='midinet_model', type=str, help='the directory to save the model to')
    parser.add_argument('-d', '--dataset', type=str, help='the directory containing the dataset to train on')
    parser.add_argument('--restore', action='store_true', help='restore from the last checkpoint')

    args = parser.parse_args()
    train(args.epochs, args.save_directory, args.dataset, args.restore)

if __name__ == "__main__":
    main()
