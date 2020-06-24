from bert_keras_layer import BertKerasModel
from bert_tokenizer import *
from sklearn.model_selection import train_test_split
import time
import argparse

if __name__ == '__main__':

    output_dir = 'bert_keras_output_{}'.format(int(time.time()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-b', default=32, type=int)
    parser.add_argument('--learning-rate', '-l', default=1e-5, type=float)
    parser.add_argument('--max-seq-length', default=128, type=int)
    parser.add_argument('--warmup-proportion', default=0.1, type=float)
    parser.add_argument('--dropout-rate', default=0.5, type=float)
    parser.add_argument('--num-train-epochs', '-n', default=3, type=int)
    parser.add_argument('--dense-size', '-d', default=256, type=int)
    parser.add_argument('--save-summary-every', default=100, type=int)
    parser.add_argument('--output-dir', '-o', default=output_dir, type=str)
    parser.add_argument('--training', '-t', default=True, type=bool)
    parser.add_argument('--bert-dir', default='uncased_L-12_H-768_A-12', type=str)
    args = parser.parse_args()

    tokenizer = create_tokenizer(args.bert_dir)

    df = get_filtered_nyt_data_with_scores('nyt_articles_with_normalized_scores.pkl').sample(100)
    df['category_labels'] = df['section'].astype('category').cat.codes
    num_classes = df['category_labels'].max() + 1
    train_df, test_df = train_test_split(df, random_state=42)
    train_ids, train_labels = tokenize_data(train_df['abstract'], train_df['category_labels'], tokenizer, args.max_seq_length, num_classes)
    test_ids, test_labels = tokenize_data(test_df['abstract'], test_df['category_labels'], tokenizer, args.max_seq_length, num_classes)
    model = BertKerasModel(num_classes, bert_dir=args.bert_dir, output_dir=args.output_dir, batch_size=args.batch_size,
        epochs=args.num_train_epochs, max_seq_length=args.max_seq_length, dense_size=args.dense_size,
        learning_rate=args.learning_rate)
    model.fit(train_ids, train_labels, test_ids, test_labels)