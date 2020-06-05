"""
Example of how to use the BERTTrainer class.
"""

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
	nyt_data = get_filtered_nyt_data('nyt_data_from_2015.pkl')
    trainer = BERTTrainer(nyt_data)
    trainer.train_model()
    eval_accuracy_info = trainer.test_model()
    prin(eval_accuracy_info)
    preds = trainer.predict([
        "Donald Trump",
        "20th century art",
        "Donald Trump likes 20th century art"
    ])
    print(preds)
    # can use these predicted values to compute various other eval metrics 
