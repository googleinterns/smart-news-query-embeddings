"""
Example of how to use the BERTTrainer class.
"""

import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from bert_trainer import BERTTrainer

def get_filtered_nyt_data(data_path):
    print('Reading data...')
    df = pd.read_pickle(data_path)
    sections = df[['section', 'desk']].drop_duplicates()
    category_counts = sections.groupby('section').count().sort_values('desk', ascending=False)
    big_category_df = category_counts[category_counts['desk'] >= 20]

    big_categories = list(big_category_df.index)

    filtered = df[df['section'].isin(big_categories)]
    return filtered

if __name__ == '__main__':
    # Read in data. No need to separate training and validation,
    # BERTTrainer class takes care of that.
    output_dir = 'bert_output_{}'.format(int(time.time()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-b', default=32, type=int)
    parser.add_argument('--learning-rate', '-l', default=2e-5, type=float)
    parser.add_argument('--max-seq-length', default=128, type=int)
    parser.add_argument('--warmup-proportion', default=0.1, type=float)
    parser.add_argument('--dropout-rate', default=0.5, type=float)
    parser.add_argument('--num-train-epochs', '-n', default=3, type=int)
    parser.add_argument('--save-checkpoints-every', default=500, type=int)
    parser.add_argument('--save-summary-every', default=100, type=int)
    parser.add_argument('--output-dir', '-o', default=output_dir, type=str)
    parser.add_argument('--training', '-t', default=True, type=bool)
    args = parser.parse_args()

    nyt_data = get_filtered_nyt_data('nyt_data_from_2015.pkl')
    trainer = BERTTrainer(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        warmup_proportion=args.warmup_proportion,
        num_train_epochs=args.num_train_epochs,
        save_checkpoints_every=args.save_checkpoints_every,
        save_summary_every=args.save_summary_every,
        dropout_rate=args.dropout_rate,
        output_dir=args.output_dir,
        is_training=args.training,
    )
    X_train, X_test, Y_train, Y_test = train_test_split(nyt_data['abstract'], nyt_data['section'], test_size=0.2, random_state=42)
    trainer.train(X_train, Y_train)
    preds = trainer.predict(X_test)
    for x, y, z in zip(X_test, preds, Y_test):
        print('Input: {}'.format(x))
        print('Prediction: {}'.format(y))
        print('Actual: {}'.format(z))
    print(np.mean(np.array(preds) == np.array(Y_test)))

    # can use these predicted values to compute various other eval metrics 
